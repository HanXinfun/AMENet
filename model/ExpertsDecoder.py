#### hjx Expert Decoder ####
from torch.distributions import Dirichlet
import torch
import torch.nn as nn
import numpy as np

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ME_DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_channels=0,
                 num_experts=4,
                 use_batchnorm=True):
        """
        多专家动态路由解码器模块。
        
        参数：
            in_channels: 输入特征的通道数（上采样分支）。
            out_channels: 输出特征的通道数。
            skip_channels: 跳跃连接（skip connection）特征的通道数。
            num_experts: 专家数量，即并行卷积分支的数目。
            use_batchnorm: 是否使用批归一化。
        """
        super().__init__()
        self.num_experts = num_experts
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # 定义多个专家，每个专家由两层卷积构成
        self.experts = nn.ModuleList([
            nn.Sequential(
                Conv2dReLU(
                    in_channels + skip_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    use_batchnorm=use_batchnorm,
                ),
                Conv2dReLU(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    use_batchnorm=use_batchnorm,
                )
            )
            for _ in range(num_experts)
        ])
        
        # 动态路由模块：通过全局池化和全连接层生成每个专家的权重
        routing_input_channels = in_channels + skip_channels
        self.routing_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 全局平均池化，输出形状 (B, routing_input_channels, 1, 1)
            nn.Flatten(),              # 展平为 (B, routing_input_channels)
            nn.Linear(routing_input_channels, num_experts),
            nn.Softmax(dim=1)          # 使用 softmax 获得归一化的权重
        )
        
    def forward(self, x, skip=None):
        # 上采样分支特征
        x_up = self.up(x)
        # 如果存在跳跃连接
        if skip is not None:
            x_combined = torch.cat([x_up, skip], dim=1)
        else:
            x_combined = x_up
        
        # 通过动态路由模块计算每个专家的权重，形状为 (B, num_experts)
        routing_weights = self.routing_fc(x_combined)
        
        # 每个专家分别处理输入特征
        expert_outputs = [expert(x_combined) for expert in self.experts]
        # 将所有专家的输出堆叠，形状变为 (B, num_experts, out_channels, H, W)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 将路由权重扩展维度，以便与专家输出相乘
        routing_weights = routing_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, num_experts, 1, 1, 1)
        
        # 对所有专家的输出进行加权求和，得到最终输出
        out = (expert_outputs * routing_weights).sum(dim=1)
        return out


        
#### hjx Expert Decoder ####
    



class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
