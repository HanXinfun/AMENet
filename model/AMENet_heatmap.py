import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from model.VSSBlock_mul import CMAMBlock
from model.ExpertsDecoder import ME_DecoderBlock, DecoderBlock


class AMENet(nn.Module):
    def __init__(self, num_classes=2):
        super(AMENet, self).__init__()
        

        resnet_rgb  = torchvision.models.resnet101(pretrained=True)
        resnet_gray = torchvision.models.resnet101(pretrained=True)

        
        # RGB 分支
        self.rgb_layer0 = nn.Sequential(
            resnet_rgb.conv1,
            resnet_rgb.bn1,
            resnet_rgb.relu,
            resnet_rgb.maxpool
        )
        self.rgb_layer1 = resnet_rgb.layer1  # out: 256
        self.rgb_layer2 = resnet_rgb.layer2  # out: 512
        self.rgb_layer3 = resnet_rgb.layer3  # out: 1024
        self.rgb_layer4 = resnet_rgb.layer4  # out: 2048

        # Gray 分支
        self.gray_layer0 = nn.Sequential(
            resnet_gray.conv1,
            resnet_gray.bn1,
            resnet_gray.relu,
            resnet_gray.maxpool
        )
        self.gray_layer1 = resnet_gray.layer1  # out: 256
        self.gray_layer2 = resnet_gray.layer2  # out: 512
        self.gray_layer3 = resnet_gray.layer3  # out: 1024
        self.gray_layer4 = resnet_gray.layer4  # out: 2048
        
        # -----------------------------------------------------
        # 2) CMAMBlock: 融合两路特征
        # -----------------------------------------------------
        self.cmam4 = CMAMBlock(hidden_dim=8*8, drop_path=0.1, attn_drop_rate=0.1, d_state=16, expand=2.0, is_light_sr=False, num_heads=16, ffn_drop=0.1)
        
        # -----------------------------------------------------
        # 3) ME-Decoder 部分 (多尺度解码)
        # -----------------------------------------------------
        self.decoder4 = DecoderBlock(in_channels=2048, skip_channels=1024,    out_channels=1024)   # 最深层, 无 skip
        self.decoder3 = DecoderBlock(in_channels=1024,  skip_channels=512, out_channels=512)
        self.decoder2 = DecoderBlock(in_channels=512,  skip_channels=256,  out_channels=256)
        self.decoder1 = ME_DecoderBlock(in_channels=256,  skip_channels=0,  out_channels=128)
        
        # -----------------------------------------------------
        # 4) 最终的分割头
        # -----------------------------------------------------
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        self.aux_seg_head = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, x_rgb, x_gray):
        # —— 1) PET 分支扩成 3 通道 —— #
        x_gray = x_gray.unsqueeze(1).repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]

        # —— 2) ResNet 主干提取多层特征 —— #
        x0_rgb  = self.rgb_layer0(x_rgb)
        x0_gray = self.gray_layer0(x_gray)

        x1_rgb  = self.rgb_layer1(x0_rgb)
        x1_gray = self.gray_layer1(x0_gray)
        c1 = x1_rgb + x1_gray

        x2_rgb  = self.rgb_layer2(x1_rgb)
        x2_gray = self.gray_layer2(x1_gray)
        c2 = x2_rgb + x2_gray

        x3_rgb  = self.rgb_layer3(x2_rgb)
        x3_gray = self.gray_layer3(x2_gray)
        c3 = x3_rgb + x3_gray

        # —— 3) 第四层 & CMAM 融合 —— #
        x4_rgb  = self.rgb_layer4(x3_rgb)    # 用于 heatmap1
        x4_gray = self.gray_layer4(x3_gray)
        b, c, H4, W4 = x4_rgb.shape
        tok_rgb  = x4_rgb.reshape(b, c, -1)
        tok_gray = x4_gray.reshape(b, c, -1)
        mamba_x  = self.cmam4(tok_rgb, tok_gray, (32, 64))
        c4       = mamba_x.reshape(b, c, H4, W4)  # 用于 heatmap2

        # —— 4) Decoder 解码过程 —— #
        d4 = self.decoder4(c4, skip=c3)
        d3 = self.decoder3(d4, skip=c2)
        d2 = self.decoder2(d3, skip=c1)           # 用于 heatmap3
        d1 = self.decoder1(d2, skip=None)

        # 上采样到原始大小
        d1_up  = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        d2_aux = F.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=False)

        # —— 5) 分割头 —— #
        logits  = self.seg_head(d1_up)
        _aux    = self.aux_seg_head(d2_aux)  # 可用于辅助损失，最终不输出

        # —— 6) 准备 Grad-CAM 特征与钩子 —— #
        feat1 = x4_rgb   # heatmap1 来源
        feat2 = c4       # heatmap2 来源
        feat3 = d1       # heatmap3 来源

        grads = {}
        # 在原始特征上打开梯度追踪并注册 hook
        feat1.requires_grad_(True)
        feat1.register_hook(lambda grad, name='g1': grads.setdefault(name, grad))
        feat2.requires_grad_(True)
        feat2.register_hook(lambda grad, name='g2': grads.setdefault(name, grad))
        feat3.requires_grad_(True)
        feat3.register_hook(lambda grad, name='g3': grads.setdefault(name, grad))

        # —— 7) 选取一个目标 logit 并反向传播 —— #
        B, NC, Hout, Wout = logits.shape
        cls_idx = 5                          # 选取类别  Vaihingen:4(car) / Potsdam:3(tree) / CBLD:1(泥流) / whu:5(road)
        y0, x0 = Hout // 2, Wout // 2
        target = logits[:, cls_idx, y0, x0]  # 中心位置
        self.zero_grad()
        target.backward(retain_graph=True)

        # —— 8) 计算 Grad-CAM —— #
        def compute_cam(feat, grad):
            # feat, grad: [B, C, H, W]
            weights = grad.mean(dim=(2,3), keepdim=True)   # [B,C,1,1]
            cam_map = (weights * feat).sum(dim=1)          # [B, H, W]
            cam_map = torch.relu(cam_map)                  # ReLU
            # Batch 内归一化到 [0,1]
            mins = cam_map.view(B, -1).min(dim=1)[0].view(B,1,1)
            maxs = cam_map.view(B, -1).max(dim=1)[0].view(B,1,1)
            return (cam_map - mins) / (maxs - mins + 1e-6)

        cam1 = compute_cam(feat1, grads['g1'])
        cam2 = compute_cam(feat2, grads['g2'])
        cam3 = compute_cam(feat3, grads['g3'])

        # —— 9) 转为 NumPy 并返回 —— #
        heatmap1 = cam1.detach().cpu().numpy()
        heatmap2 = cam2.detach().cpu().numpy()
        heatmap3 = cam3.detach().cpu().numpy()

        # —— 去掉 Batch 维度 —— #
        # 假设 B>=1，我们只取第一个样本
        # logits    = logits[0]      # [C, H, W]
        heatmap1  = heatmap1[0]    # [H1, W1]
        heatmap2  = heatmap2[0]    # [H2, W2]
        heatmap3  = heatmap3[0]    # [H3, W3]

        return logits, heatmap1, heatmap2, heatmap3


    # tsne 用
    def extract_features(self, x_rgb, x_gray):
        """
        提取 AMENet 分割头（seg_head）之前的全局特征向量。
        在最后一层解码器输出 d1 的空间维度上做全局平均池化，得到 [B, C] 的特征。

        参数:
            x_rgb: Tensor, 形状 [B, 3, H, W]，RGB 图像输入
            x_gray: Tensor, 形状 [B, H, W]，单通道 DSM 输入

        返回:
            features: Tensor，形状 [B, C]，C 为 d1 通道数（128）
        """

        # —— 1) DSM 分支拓展到 3 通道 —— #
        # x_gray 原来 [B, H, W]，先 unsqueeze 再重复
        x_gray = x_gray.unsqueeze(1)        # [B, 1, H, W]
        x_gray = x_gray.repeat(1, 3, 1, 1)   # [B, 3, H, W]

        # —— 2) 两条 ResNet 主干提取多层特征 —— #
        # 第 0 层
        x0_rgb  = self.rgb_layer0(x_rgb)    # [B,  64, H/4,  W/4]
        x0_gray = self.gray_layer0(x_gray)  # [B,  64, H/4,  W/4]

        # 第 1 层
        x1_rgb  = self.rgb_layer1(x0_rgb)   # [B, 256, H/4,  W/4]
        x1_gray = self.gray_layer1(x0_gray) # [B, 256, H/4,  W/4]
        c1 = x1_rgb + x1_gray               # 融合 RGB + DSM 特征

        # 第 2 层
        x2_rgb  = self.rgb_layer2(x1_rgb)   # [B, 512, H/8,  W/8]
        x2_gray = self.gray_layer2(x1_gray) # [B, 512, H/8,  W/8]
        c2 = x2_rgb + x2_gray

        # 第 3 层
        x3_rgb  = self.rgb_layer3(x2_rgb)   # [B,1024, H/16, W/16]
        x3_gray = self.gray_layer3(x2_gray) # [B,1024, H/16, W/16]
        c3 = x3_rgb + x3_gray

        # —— 3) 第 4 层 & CMAM 融合 —— #
        x4_rgb  = self.rgb_layer4(x3_rgb)   # [B,2048, H/32, W/32]
        x4_gray = self.gray_layer4(x3_gray) # [B,2048, H/32, W/32]
        b, c, H4, W4 = x4_rgb.shape

        # 展平通道和空间，输入到 CMAM4 模块
        tok_rgb  = x4_rgb.reshape(b, c, -1)   # [B,2048, N]
        tok_gray = x4_gray.reshape(b, c, -1)  # [B,2048, N]
        mamba_x  = self.cmam4(tok_rgb, tok_gray, (32, 64))  # [B,2048, 32*64?]
        c4       = mamba_x.reshape(b, c, H4, W4)            # [B,2048, H/32, W/32]

        # —— 4) 解码器多尺度解码 —— #
        d4 = self.decoder4(c4, skip=c3)      # [B,1024, H/16, W/16]
        d3 = self.decoder3(d4, skip=c2)      # [B, 512, H/8,  W/8]
        d2 = self.decoder2(d3, skip=c1)      # [B, 256, H/4,  W/4]
        d1 = self.decoder1(d2, skip=None)    # [B, 128, H/2,  W/2]

        # —— 5) 全局平均池化，得到 [B,128] 的特征向量 —— #
        features = d1.mean(dim=(2, 3))       # 对 H,W 两个维度做平均
        return features


# -----------------------------------------------
#  python AMENet.py
# -----------------------------------------------
from torchinfo import summary
import time
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 生成输入数据
    x_rgb = torch.randn(1, 3, 256, 256).to(device)
    x_gray = torch.randn(1, 256, 256).to(device) 

    # 初始化模型
    model = AMENet(num_classes=6).to(device)

    # 验证输出形状
    y,_,_,_ = model(x_rgb, x_gray)
    print("Output shape:", y.shape)
