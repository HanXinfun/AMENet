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
        

        # (灰度图 repeat 成 3 通道, 这样可以直接用官方预训练的resnet50)
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
        

        self.decoder4 = DecoderBlock(in_channels=2048, skip_channels=1024,    out_channels=1024)   # 最深层, 无 skip
        self.decoder3 = DecoderBlock(in_channels=1024,  skip_channels=512, out_channels=512)
        self.decoder2 = DecoderBlock(in_channels=512,  skip_channels=256,  out_channels=256)
        # -----------------------------------------------------
        # 3) ME-Decoder 部分 (多尺度解码)
        # -----------------------------------------------------
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
        """
        x_rgb: [B, 3, H, W]
        x_gray: [B, H, W] (做法B时，会先repeat到3通道)
        """
        
        x_gray = x_gray.unsqueeze(1)        # [B,H,W] -> [B,1,H,W]
        x_gray = x_gray.repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]
        
        # -----------------------
        # 1) 分别通过两条ResNet
        # -----------------------
        # layer0
        x0_rgb  = self.rgb_layer0(x_rgb)   # [B, 64,  H/4,  W/4]
        x0_gray = self.gray_layer0(x_gray) # [B, 64,  H/4,  W/4]
        
        # layer1
        x1_rgb  = self.rgb_layer1(x0_rgb)   # [B, 256, H/4,  W/4]
        x1_gray = self.gray_layer1(x0_gray) # [B, 256, H/4,  W/4]
        # # add 融合
        c1 = x1_rgb + x1_gray
        
        # layer2
        x2_rgb  = self.rgb_layer2(x1_rgb)   # [B, 512, H/8,  W/8]
        x2_gray = self.gray_layer2(x1_gray) # [B, 512, H/8,  W/8]
        # # add 融合
        c2 = x2_rgb + x2_gray
        
        # layer3
        x3_rgb  = self.rgb_layer3(x2_rgb)   # [B, 1024,H/16, W/16]
        x3_gray = self.gray_layer3(x2_gray) # [B, 1024,H/16, W/16]
        # # add 融合
        c3 = x3_rgb + x3_gray
        
        # layer4
        x4_rgb  = self.rgb_layer4(x3_rgb)   # [B, 2048,H/32, W/32]
        x4_gray = self.gray_layer4(x3_gray) # [B, 2048,H/32, W/32]
        # CMAM 融合
        b, c, w, h = x4_rgb.shape
        x4_rgb_token = x4_rgb.reshape(b, c, -1)
        x4_gray_token = x4_gray.reshape(b, c, -1)
        mamba_x = self.cmam4(x4_rgb_token, x4_gray_token, (32, 64)) # [b, 2048, 8*8]
        c4 = mamba_x.reshape(b, c, w, h)    # [B, 2048,H/32, W/32]  
        


        # -----------------------
        # 2) Decoder 部分 (多尺度)
        #    逐级与 c3, c2, c1 做 skip
        # -----------------------

        d4 = self.decoder4(c4, skip=c3)         # [B, 1024,  H/16, W/16]
        d3 = self.decoder3(d4, skip=c2)         # [B, 512,  H/8, W/8]
        d2 = self.decoder2(d3, skip=c1)         # [B, 256,  H/4,  W/4]
        d1 = self.decoder1(d2, skip=None)         # [B, 128,   H/2,  W/2]
        
        # 上采样回到原图大小 (H, W)
        d1_up = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)  # [B,128,H,W]
        # 辅助损失使用
        d2_aux = F.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=False)  # [B,256,H,W]
        
        # -----------------------
        # 3) 分割头
        # -----------------------
        out = self.seg_head(d1_up)  # [B, num_classes, H, W]
        # 辅助损失使用
        out_aux = self.aux_seg_head(d2_aux) # # [B, num_classes, H, W]
        
        return out, out_aux

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
    
    # 计算Params和FLOPs
    stats = summary(model, input_data=[x_rgb, x_gray], verbose=0)
    print(f"Total Params: {stats.total_params / 1e6:.2f}M")
    print(f"Total FLOPs: {stats.total_mult_adds / 1e9:.2f}G")

    # 计算内存占用
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x_rgb, x_gray)
    memory_mb = torch.cuda.max_memory_allocated(device=device) / 1e6
    print(f"Memory Usage: {memory_mb:.2f} MB")

    # 计算推理速度
    with torch.no_grad():
        for _ in range(10):  # 预热
            _ = model(x_rgb, x_gray)
    repeat = 100
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(x_rgb, x_gray)
    torch.cuda.synchronize()
    end_time = time.time()
    fps = repeat / (end_time - start_time)
    print(f"Speed: {fps:.2f} FPS")

    # 验证输出形状
    y = model(x_rgb, x_gray)
    print("Output shape:", y.shape)
