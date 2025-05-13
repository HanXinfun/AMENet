# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from .vmamba import VSSM
# from vmamba import VSSM

logger = logging.getLogger(__name__)

class MambaUNet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(MambaUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.mamba_unet =  VSSM(
                                patch_size=config.MODEL.VSSM.PATCH_SIZE,
                                in_chans=config.MODEL.VSSM.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.VSSM.EMBED_DIM,
                                depths=config.MODEL.VSSM.DEPTHS,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.mamba_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")








# -----------------------------------------------
#  python MambaUNet.py
# -----------------------------------------------
from torchinfo import summary
import torch
import time

if __name__ == '__main__':
    import argparse
    from MambaUNet_config import get_config

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MambaUnet for Image Segmentation")
    parser.add_argument('--cfg', type=str, default="vmamba_tiny.yaml", help='path to config file')
    args = parser.parse_args()

    # 加载配置文件
    config = get_config(args)
    
    # 初始化模型并移动到 GPU
    device = torch.device("cuda:1")  # 确保与输入设备一致
    model = MambaUNet(config, img_size=256, num_classes=6).to(device)
    model.eval()

    # 生成输入数据 (假设输入为单模态 RGB 图像)
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    # ---------------------- 计算 Params 和 FLOPs ----------------------
    stats = summary(model, input_data=input_tensor, verbose=0)  # 单输入模型无需列表
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
