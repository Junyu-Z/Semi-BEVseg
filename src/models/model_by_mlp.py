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
from operator import mul
from functools import reduce
import math
from torchvision.transforms.functional import rotate, resize, pad
import torchvision
import warnings
warnings.filterwarnings("ignore")

from fpn import FPN50
from resnet import ResNetLayer


class Resampler(nn.Module):

    def __init__(self, resolution, extents):
        super().__init__()

        # Store z positions of the near and far planes
        self.near = extents[1]
        self.far = extents[3]

        # Make a grid in the x-z plane
        self.grid = _make_grid(resolution, extents)


    def forward(self, features, calib):

        # Copy grid to the correct device
        self.grid = self.grid.to(features)
        
        # We ignore the image v-coordinate, and assume the world Y-coordinate
        # is zero, so we only need a 2x2 submatrix of the original 3x3 matrix
        calib = calib[:, [0, 2]][..., [0, 2]].view(-1, 1, 1, 2, 2)

        # Transform grid center locations into image u-coordinates
        cam_coords = torch.matmul(calib, self.grid.unsqueeze(-1)).squeeze(-1)

        # Apply perspective projection and normalize
        ucoords = cam_coords[..., 0] / cam_coords[..., 1]
        ucoords = ucoords / features.size(-1) * 2 - 1

        # Normalize z coordinates
        zcoords = (cam_coords[..., 1]-self.near) / (self.far-self.near) * 2 - 1

        # Resample 3D feature map
        grid_coords = torch.stack([ucoords, zcoords], -1).clamp(-1.1, 1.1)
        return F.grid_sample(features, grid_coords)


def _make_grid(resolution, extents):
    # Create a grid of cooridinates in the birds-eye-view
    x1, z1, x2, z2 = extents
    zz, xx = torch.meshgrid(
        torch.arange(z1, z2, resolution), torch.arange(x1, x2, resolution))

    return torch.stack([xx, zz], dim=-1)


class DenseTransformer(nn.Module):

    def __init__(self, in_channels=256, channels=64, resolution=0.25, grid_extents=[-25.0, 1.0, 25.0, 2.25], 
                 ymin=-2, ymax=4, focal_length=630, groups=1):
        super().__init__()

        # Initial convolution to reduce feature dimensions
        self.conv = nn.Conv2d(in_channels, channels, 1)
        self.bn = nn.GroupNorm(16, channels)

        # Resampler transforms perspective features to BEV
        self.resampler = Resampler(resolution, grid_extents)

        # Compute input height based on region of image covered by grid
        self.zmin, zmax = grid_extents[1], grid_extents[3]  # 
        self.in_height = math.ceil(focal_length * (ymax - ymin) / self.zmin)  # range of row
        self.ymid = (ymin + ymax) / 2  # 1

        # Compute number of output cells required
        self.out_depth = math.ceil((zmax - self.zmin) / resolution)

        # Dense layer which maps UV features to UZ
        self.fc = nn.Conv1d(
            channels * self.in_height, channels * self.out_depth, 1, groups=groups
        )
        self.out_channels = channels

    def forward(self, features, calib, *args):
        # Crop feature maps to a fixed input height
        features = torch.stack([self._crop_feature_map(fmap, cal) 
                                for fmap, cal in zip(features, calib)])
        
        # Reduce feature dimension to minimize memory usage
        features = F.relu(self.bn(self.conv(features)))

        # Flatten height and channel dimensions
        B, C, _, W = features.shape
        flat_feats = features.flatten(1, 2)
        bev_feats = self.fc(flat_feats).view(B, C, -1, W)

        # Resample to orthographic grid
        return self.resampler(bev_feats, calib)


    def _crop_feature_map(self, fmap, calib):
        
        # Compute upper and lower bounds of visible region
        focal_length, img_offset = calib[1, 1:]  # fy, cy
        vmid = self.ymid * focal_length / self.zmin + img_offset
        vmin = math.floor(vmid - self.in_height / 2)
        vmax = math.floor(vmid + self.in_height / 2)

        # Pad or crop input tensor to match dimensions
        return F.pad(fmap, [0, 0, -vmin, vmax - fmap.shape[-2]])


class TransformerPyramid(nn.Module):
    def __init__(self, in_channels=256, channels=64, resolution=0.25, extents=[-25., 1., 25., 50.],
                 ymin=-2, ymax=4, focal_length=630):
        super().__init__()
        self.transformers = nn.ModuleList()
        for i in range(5):  # 0,1,2,3,4
            # Scaled focal length for each transformer
            focal = focal_length / pow(2, i + 3)  # 78.75, 39.375, 19.6875, 9.84375, 4.921875
            # Compute grid bounds for each transformer
            zmax = min(math.floor(focal * 2) * resolution, extents[3])      # 39.25, 19.50, 9.75, 4.75, 2.25
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]  # 19.50,  9.75, 4.75, 2.25, 1.00
            subset_extents = [extents[0], zmin, extents[2], zmax]
            # print(subset_extents, ymin, ymax)
            # Build transformers
            tfm = DenseTransformer(in_channels, channels, resolution, subset_extents, ymin, ymax, focal)
            self.transformers.append(tfm)

    def forward(self, feature_maps, calib):  # 1x256x57x100, 1x256x29x50, 1x256x15x25, 1x256x8x13, 1x256x4x7
        bev_feats = list()
        for i, fmap in enumerate(feature_maps):
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2 ** i  # 8, 16, 32, 64, 128
            calib_downsamp = calib.clone()
            calib_downsamp[:, :2] = calib[:, :2] / scale

            # Apply orthographic transformation to each feature map separately
            bev_feats.append(self.transformers[i](fmap, calib_downsamp))

        # Combine birds-eye-view feature maps along the depth axis(row axis)
        return torch.cat(bev_feats[::-1], dim=-2)


class TopdownNetwork(nn.Sequential):
    def __init__(self, in_channels=64, channels=128, layers=[4, 4], 
                 strides=[1, 2], blocktype='bottleneck'):

        modules = list()
        self.downsample = 1
        for nblocks, stride in zip(layers, strides):
            # print(in_channels, ',', channels)  # 64,128; 512,64
            # Add a new residual layer
            module = ResNetLayer(
                in_channels, channels, nblocks, 1/stride, blocktype=blocktype)
            modules.append(module)

            # Halve the number of channels at each layer
            in_channels = module.out_channels
            channels = channels // 2
            self.downsample *= stride
        
        self.out_channels = in_channels

        super().__init__(*modules)


class model_by_mlp(nn.Module):
    def __init__(self, config):
        super(model_by_mlp, self).__init__()
        self.frontend = FPN50()
        tfm_resolution = config.map_resolution * reduce(mul, config.topdown.strides)
        self.transformer = TransformerPyramid(256, config.tfm_channels, tfm_resolution,
                                              config.map_extents, config.ymin, 
                                              config.ymax, config.focal_length)
        self.topdown = TopdownNetwork(config.tfm_channels, config.topdown.channels,
                                      config.topdown.layers, config.topdown.strides,
                                      config.topdown.blocktype)
        self.classifier = nn.Conv2d(self.topdown.out_channels, config.num_class, 1)

    def forward(self, img, K):  # B x 3 x H x W
        # Extract multiscale feature maps
        feature_maps = self.frontend(img)
        # Transform image features to birds-eye-view
        bev_feature = self.transformer(feature_maps, K)  # bx64x98x100
        # Apply topdown network
        td_feats = self.topdown(bev_feature)  # bx256x196x200
        # Predict individual class log-probabilities
        bev_seg = self.classifier(td_feats)  # bxnx196x200
        
        return bev_feature, bev_seg


from yacs.config import CfgNode

if __name__ == '__main__':    
    with open('/home/zjy/workspace/semi-BEVseg/configs/config.yml') as f:
        config = CfgNode.load_cfg(f)
    config['num_class'] = 14
    
    bev_seg_model = model_by_mlp(config).cuda()
    
    print('num of trainable parameters =', sum(p.numel() for p in bev_seg_model.parameters() if p.requires_grad))
    
    img = torch.ones((2, 3, 600, 800)).cuda()  # 1 x 3 x 224 x 448
    K = torch.tensor([[[300.0, 0, 400], [0, 300, 300], [0, 0, 1]], [[300, 0, 400], [0, 300, 300], [0, 0, 1]]]).cuda()
    
    bev_seg, _ = bev_seg_model(img, K)  # 1 x 14 x X x Y
    
    print(bev_seg.shape)
 

