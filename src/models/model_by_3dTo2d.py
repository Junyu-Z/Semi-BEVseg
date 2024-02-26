"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18


class CamEncoder(nn.Module):
    def __init__(self, backbone_type='b0'):  # D==41, C==64
        super(CamEncoder, self).__init__()

        if backbone_type == 'b0':
            self.trunk = EfficientNet.from_name("efficientnet-b0")
            pretrained = torch.load('/home/ma-user/modelarts/user-job-dir/BEVmix/BEVmix/src/models/efficientnet-b0-355c32eb.pth')
            self.trunk.load_state_dict(pretrained)
            self.up1 = Up(320 + 112, 512)
        elif backbone_type == 'b4':
            self.trunk = EfficientNet.from_name("efficientnet-b4")
            pretrained = torch.load('/home/ma-user/modelarts/user-job-dir/BEVmix/BEVmix/src/models/efficientnet-b4-6ed6700e.pth')
            self.trunk.load_state_dict(pretrained)
            self.up1 = Up(448 + 160, 512)

        self.conv = nn.Conv2d(512, 64, kernel_size=1, padding=0)

    def get_cam_feat(self, x):  # bx3xhxw
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        # print(endpoints.keys())  # 'reduction_1', 'reduction_2', 'reduction_3', 'reduction_4', 'reduction_5'
        # print(endpoints['reduction_4'].shape)  # b x 112 x h/16 x w/16
        # print(endpoints['reduction_5'].shape)  # b x 320 x h/32 x w/32
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])  # bx512xh/16xw/16
        x = self.conv(x)
        
        return x

    def forward(self, x):  # b x 3 x h x w
        f = self.get_cam_feat(x)  # b x 512 x h/16 x w /16

        return f


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',  align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class ViewTransform(nn.Module):
    def __init__(self):
        super(ViewTransform, self).__init__()
        x = torch.arange(-25+0.125, 25, 0.5)  # -24.875 ~ +24.875, size = 100
        z = torch.arange(0+0.125, 50, 0.5)  # +0.125 ~ +49.875, size = 100
        z, x = torch.meshgrid(z, x, indexing='ij')
        y1 = -1.0 * torch.ones((100, 100))
        y2 = +0.5 * torch.ones((100, 100))
        y3 = +2.0 * torch.ones((100, 100))

        xz  = (x / z).unsqueeze(0)   # b x 100 x 100
        yz1 = (y1 / z).unsqueeze(0)  # b x 100 x 100
        yz2 = (y2 / z).unsqueeze(0)  # b x 100 x 100
        yz3 = (y3 / z).unsqueeze(0)  # b x 100 x 100
        
        self.xz  = nn.Parameter(xz,  requires_grad=False)  # b x 100 x 100
        self.yz1 = nn.Parameter(yz1, requires_grad=False)  # b x 100 x 100
        self.yz2 = nn.Parameter(yz2, requires_grad=False)  # b x 100 x 100
        self.yz3 = nn.Parameter(yz3, requires_grad=False)  # b x 100 x 100
        
    def forward(self, img_feature, img_size, intrinsic):  # b x 64 x h x w, b x 3 x 3  -->  b x 64 x 100 x 100
        b = img_feature.shape[0]
        w, h = img_size
        
        xz  = self.xz.repeat(b, 1, 1)
        yz1 = self.yz1.repeat(b, 1, 1)
        yz2 = self.yz2.repeat(b, 1, 1)
        yz3 = self.yz3.repeat(b, 1, 1)
        
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).repeat(1, 100, 100)  # b x 100 x 100
        cx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).repeat(1, 100, 100)  # b x 100 x 100
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).repeat(1, 100, 100)  # b x 100 x 100
        cy = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).repeat(1, 100, 100)  # b x 100 x 100
        
        u1 = ((fx * xz + cx) / w - 0.5) * 2  # b x 100 x 100
        v1 = ((fy * yz1 + cy) / h - 0.5) * 2 # b x 100 x 100
        grid1 = torch.stack((u1, v1), dim=3)  # b x 100 x 100 x 2, -1.0m plane
        
        u2 = ((fx * xz + cx) / w - 0.5) * 2  # b x 100 x 100
        v2 = ((fy * yz2 + cy) / h - 0.5) * 2  # b x 100 x 100
        grid2 = torch.stack((u2, v2), dim=3)  # b x 100 x 100 x 2, +0.5m plane
        
        u3 = ((fx * xz + cx) / w - 0.5) * 2  # b x 100 x 100
        v3 = ((fy * yz3 + cy) / h - 0.5) * 2  # b x 100 x 100
        grid3 = torch.stack((u3, v3), dim=3)  # b x 100 x 100 x 2,  +2.0m plane
        
        bev_feature1 = torch.nn.functional.grid_sample(img_feature, grid1, mode='bilinear', padding_mode='zeros', align_corners=False)  # b x 64 x 100 x 100
        bev_feature2 = torch.nn.functional.grid_sample(img_feature, grid2, mode='bilinear', padding_mode='zeros', align_corners=False)
        bev_feature3 = torch.nn.functional.grid_sample(img_feature, grid3, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        bev_feature = (bev_feature1 + bev_feature2 + bev_feature3) / 3  # b x 64 x 100 x 100
        
        return bev_feature


class BevEncoder(nn.Module):
    def __init__(self, in_channels=64, outC=14):
        super(BevEncoder, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):  # x: B x 64 x 100 x 100
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv1(x)  # x: B x 64 x 100 x 100
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # x1: B x 64 x 100 x 100
        x = self.layer2(x1)  # x: B x 128 x 50 x 50
        x = self.layer3(x)  # x: B x 256 x 25 x 25

        x = self.up1(x, x1)  # 给x进行4倍上采样然后和x1 concat 在一起  x: B x 256 x 100 x 100
        x = self.up2(x)  # 2倍上采样->3x3卷积->1x1卷积  x: B x outC x 200 x 200
        x = x[..., 4:, :]  # B x outC x 196 x 200

        return x  # B x outC x 196 x 200


class model_by_3dTo2d(nn.Module):
    def __init__(self, config):
        super(model_by_3dTo2d, self).__init__()
        self.img_size = config.img_size
        self.img_encoder = CamEncoder('b4')
        self.vt = ViewTransform()
        self.bev_encoder = BevEncoder(in_channels=64, outC=config.num_class)

    def forward(self, img, K):  # B x 3 x H x W
        img_feature = self.img_encoder(img)  # B x 64 x H/16 x W/16
        # print(img_feature.shape)
        bev_feature = self.vt(img_feature, self.img_size, K)  # B x 64 x 100 x 100
        # print(bev_feature.shape)
        bev_seg = self.bev_encoder(bev_feature)  # B x n x 196 x 200
        # print(bev_seg.shape)
        return bev_feature, bev_seg


if __name__ == '__main__':    
    bev_sef_model = model_by_3dTo2d(config).cuda()
    
    print('num of trainable parameters =', sum(p.numel() for p in bev_sef_model.parameters() if p.requires_grad))
    
    img = torch.ones((2, 3, 224, 448)).cuda()  # 1 x 3 x 224 x 448
    K = torch.tensor([[[300, 0, 224], [0, 300, 112], [0, 0, 1]], [[300, 0, 224], [0, 300, 112], [0, 0, 1]]]).cuda()
    
    bev_feature, bev_seg = bev_sef_model(img, K)  # 1 x 14 x X x Y
    
    
    

