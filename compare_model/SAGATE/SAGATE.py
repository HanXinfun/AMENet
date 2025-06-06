# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict


# from dual_resnet import resnet101
from compare_model.SAGATE.dual_resnet import resnet101


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, bn_momentum=0.003):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)



class SAGATE(nn.Module):
    def __init__(self, out_planes, criterion, norm_layer, pretrained_model=None):
        super(SAGATE, self).__init__()
        self.backbone = resnet101(pretrained_model, norm_layer=norm_layer,
                                bn_eps=1e-5,
                                bn_momentum=0.1,
                                deep_stem=True, stem_width=64)
        self.dilate = 2

        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.head = Head(out_planes, norm_layer, bn_momentum=0.1)

        self.business_layer = []
        self.business_layer.append(self.head)
        self.criterion = criterion


    def forward(self, data, hha, label=None):

        hha = hha.unsqueeze(1)        # [B,H,W] -> [B,1,H,W]
        hha = hha.repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]

        b, c, h, w = data.shape
        blocks, merges = self.backbone(data, hha)
        pred, aux_fm = self.head(merges)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        aux_fm = F.interpolate(aux_fm, size=(h, w), mode='bilinear',  align_corners=True)

        if label is not None:       # training
            loss = self.criterion(pred, label)
            loss_aux = self.criterion(aux_fm, label)

            return loss, loss_aux

        return pred

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)



class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool

class Head(nn.Module):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
            )

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1)
                                       )

        self.classify = nn.Conv2d(in_channels=256, out_channels=self.classify_classes, kernel_size=1,
                                        stride=1, padding=0, dilation=1, bias=True)

        self.auxlayer = _FCNHead(2048, classify_classes, bn_momentum=bn_momentum, norm_layer=norm_act)

    def forward(self, f_list):
        f = f_list[-1]
        encoder_out = f
        f = self.aspp(f)

        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)

        f = F.interpolate(f, size=(low_h, low_w), mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        f = self.last_conv(f)

        pred = self.classify(f)

        aux_fm = self.auxlayer(encoder_out)
        return pred, aux_fm


if __name__ == '__main__':
    # from easydict import EasyDict as edict
    # config = edict()
    # config.bn_eps = 1e-5
    # config.bn_momentum = 0.1
    model = SAGATE(6, criterion=nn.CrossEntropyLoss(),
                pretrained_model=None,
                norm_layer=nn.BatchNorm2d)
    left = torch.randn(2, 3, 256, 256)
    right = torch.randn(2, 256, 256)    # -> (2, 3, 256, 256)

    out = model(left, right)
    print(out.shape)
