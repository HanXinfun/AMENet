from .vmamba import VSSM
# from vmamba import VSSM

import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.vmunet(x)
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            # pretrained_dict = modelCheckpoint['model']
            pretrained_dict = modelCheckpoint
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            # pretrained_odict = modelCheckpoint['model']
            pretrained_odict = modelCheckpoint
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")



    
    

# -----------------------------------------------
#  python VMUNet.py
# -----------------------------------------------

from torchinfo import summary
import torch
import time

if __name__ == '__main__':
    # 初始化模型并移动到 GPU
    device = torch.device("cuda:1")  # 指定设备
    model = VMUNet(
        input_channels=3,
        num_classes=6,
    ).to(device)
    model.eval()  # 设置为推理模式

    # 生成输入数据 (假设输入为单模态 RGB 图像)
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    # ---------------------- 计算 Params 和 FLOPs ----------------------
    stats = summary(model, input_data=input_tensor, verbose=0)  # 单输入模型
    print(f"Total Params: {stats.total_params / 1e6:.2f}M")
    print(f"Total FLOPs: {stats.total_mult_adds / 1e9:.2f}G")

    # ---------------------- 计算显存占用 (MB) ----------------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)
    with torch.no_grad():
        _ = model(input_tensor)
    memory_mb = torch.cuda.max_memory_allocated(device=device) / 1e6
    print(f"Memory Usage: {memory_mb:.2f} MB")

    # ---------------------- 计算推理速度 (FPS) ----------------------
    # 预热（避免 CUDA 初始化影响测速）
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # 正式测速（同步设备确保时间准确）
    repeat = 100
    torch.cuda.synchronize(device=device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(input_tensor)
    torch.cuda.synchronize(device=device)
    end_time = time.time()
    
    fps = repeat / (end_time - start_time)
    print(f"Speed: {fps:.2f} FPS")

    # 验证输出形状
    with torch.no_grad():
        logits = model(input_tensor)
    print("Logits shape:", logits.shape)
