"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from yacs.config import CfgNode
from fpn import FPN50
import math


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])  # 0.5, 0.5, 20
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])  # -49.5, -49.5, 0
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])  # 200, 200, 1

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    # x.shape == Nprime x C
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)  # all true
    kept[:-1] = (ranks[1:] != ranks[:-1])  #
    # remove redunant rank
    x, geom_feats = x[kept], geom_feats[kept]
    # collapse dimension using kept
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None

# ========================================

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

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


class CamEncode(nn.Module):
    def __init__(self, D=41, C=64):  # D==41, C==64
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.backbone = FPN50()
        self.depthnet = nn.Conv2d(256, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):  # N*B x 3 x H x W
        # x = self.get_eff_depth(x)  # N*B x 512 x H/16 x W/16
        _, x, _, _, _ = self.backbone(x)  # N*B x 256 x H/16 x W/16
        # x = x.unsuqeeze(1)
        # print('2d feature shape =', x.shape)
        # Depth
        x = self.depthnet(x)  # B x (D+C) x N x H x W

        depth = self.get_depth_dist(x[:, :self.D])  # B x D x N x H x W
        # B x 1 x D x N x H x W  * B x C x 1 x N x H x W == B x C x D x N x H x W
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)  # B x C x D x N x H x W

        return depth, new_x

    def forward(self, x):
        # depth: B*N x D x fH x fW,  x: B*N x C x D x fH x fW
        # print('x.shape =', x.shape)  # b x 3 x H x W
        depth, x = self.get_depth_feat(x)
        # print('front depth shape =', depth.shape, ', front feature shape =', x.shape) # b x 64 x H/16 x W/16, b x 14 x H/16 x W/16

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

    def forward(self, x):  # x: B x 64 x 200 x 200
        x = self.conv1(x)  # x: B x 64 x 100 x 100
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # x1: B x 64 x 100 x 100
        x = self.layer2(x1)  # x: B x 128 x 50 x 50
        x = self.layer3(x)  # x: B x 256 x 25 x 25

        x = self.up1(x, x1)  # 给x进行4倍上采样然后和x1 concat 在一起  x: B x 256 x 100 x 100
        x = self.up2(x)  # 2倍上采样->3x3卷积->1x1卷积  x: B x 1 x 200 x 200

        return x


class model_by_2dTo3d(nn.Module):
    def __init__(self, config):
        super(model_by_2dTo3d, self).__init__()
        
        self.img_size = config.img_size
        dx, bx, nx = gen_dx_bx([-25.0, 25.0, 0.25], [0.0, 50.0, 0.25], [-10.0, 10.0, 20.0])
        self.dx = nn.Parameter(dx, requires_grad=False)  # tensor([-25.0, 25.0, 0.25])  --> stride of x,y,z grids
        self.bx = nn.Parameter(bx, requires_grad=False)  # tensor([1.0, 50.0, 0.25])  --> start of x,y,x grids
        self.nx = nn.Parameter(nx, requires_grad=False)  # tensor([-10.0, 10.0, 20.0])  --> num of x,y,z grids

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()  # D x fH x fW x 3 (column, row, depth)
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(D=self.D, C=self.camC)  # D==41, camC==64
        self.bevencode = BevEncode(inC=self.camC, outC=config.num_class)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.img_size[0], self.img_size[1]
        fH, fW = math.ceil(ogfH / self.downsample), math.ceil(ogfW // self.downsample)

        ds = torch.arange(*[4.0, 45.0, 1.0], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        # [0, (ogfW - 1)/(fW - 1), ..., (ogfW - 1)] 1xfW -> 1x1xfW -> DxfHxfW
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # columns
        # [0, (ogfH - 1)/(fH - 1), ..., (ogfH - 1)] 1xfH -> 1xfHx1 -> DxfHxfW
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # rows
        # D x fH x fW x 3
        frustum = torch.stack((xs, ys, ds), -1)
        
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrins):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        # B, N, _ = trans.shape  # B x N x 3
        B = intrins.shape[0]
        
        # undo post-transformation
        # (B x N x D x H/16 x W/16 x 3) - (B x N x 1 x 1 x 1 x 3)  -->  (B x N x D x H/16 x W/16 x 3), (col, row, depth)
        points = self.frustum.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1, -1, -1) # - post_trans.view(B, N, 1, 1, 1, 3)  # post_trans[..., 2] == 0
        
        # (B x N x 1 x 1 x 1 x 3 x 3) x (B x N x D x H/16 x W/16 x 3 x 1)  -->  (B x N x D x H/16 x W/16 x 3 x 1)
        # points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = points.unsqueeze(-1)
        
        # cam_to_ego
        # (z*u,z*v,z*1)' = z * (u, v, 1)' = K * (x, y, z)'
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)  # B x N x D x H/16 x W/16 x 3 x 1
        
        # combine = rots.matmul(torch.inverse(intrins))  # B x N x 3 x 3   R * K^(-1)
        combine = torch.inverse(intrins)  # B x N x 3 x 3  K^(-1)

        # (B x N x 1 x 1 x 1 x 3 x 3) x (B x N x D x H/16 x W/16 x 3 x 1)  ->  (B x N x D x H/16 x W/16 x 3)
        points = combine.view(B, 1, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)  # R * K^(-1) * z * (u, v, 1)' = R * (x, y, z)'
        # points += trans.view(B, N, 1, 1, 1, 3)  # B x N x D x H/16 x W/16 x 3, R * (x, y, z)' + (tx, ty, tz)' == (x', y', z')'

        return points  # B x N x D x H/16 x W/16 x 3, (x', y', z')'

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B x N x 3 x H x W
        # print(x.shape)
        x = x.view(B*N, C, imH, imW)  # N*B x 3 x H x W
        # print(x.shape)
        x = self.camencode(x)  # N*B x C x D x H/16 x W/16
        # print(x.shape)
        x = x.view(B, N, self.camC, self.D, x.shape[-2], x.shape[-1])  # B x N x C x D x H/16 x W/16
        # print(x.shape)
        x = x.permute(0, 1, 3, 4, 5, 2)  # B x N x D x H/16 x W/16 x C
        # print(x.shape)
        # print('==========')

        return x  # B x N x D x H/16 x W/16 x C

    def voxel_pooling(self, geom_feats, x):  # B x N x D x H/16 x W/16 x 3, B x N x D x H/16 x W/16 x C
        B, N, D, H, W, C = x.shape  # B x N x D x H' x W' x C
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)  # (B x N x D x H' x W') x C

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # B x N x D x H' x W' x 3
        geom_feats = geom_feats.view(Nprime, 3)  # (B x N x D x H' x W') x 3
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # (B x N x D x H' x W') x 1

        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # (B x N x D x H' x W') x 4

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]  # features
        geom_feats = geom_feats[kept]  # positions

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            # BEV平面上相同位置的feature被累加在了一起，x保存最终BEV平面有效位置上的feature，geom_feats保存这些有效位置的（X,Y,Z,b）
            # X belongs to 0~200, Y belongs to 0~200, Z==0, b belongs to 0~B
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # B x C x Z(1) x X x Y
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # put the features on the BEV plane

        # collapse Z
        # unbind: 移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片
        final = torch.cat(final.unbind(dim=2), 1)  # B x C x Z x X x Y --> B x (C*Z) x X x Y, (4 x 64 x 200 x 200)

        return final

    def get_voxels(self, x, intrins):
        geom = self.get_geometry(intrins)  # B x N x D x H/16 x W/16 x 3, (x,y,z)'
        # print(geom.shape)
        # images to features
        x = self.get_cam_feats(x)  # B x N x 3 x H x W  -->  B x N x D x H/16 x W/16 x C
        # print('frustrum feature shape =', x.shape)
        x = self.voxel_pooling(geom, x)  # B x C x X x Y
        # print('bev feature shape =', x.shape)

        return x

    def forward(self, x, intrins):
        x = x.unsqueeze(1)
        intrins = intrins.unsqueeze(1)
        
        bev_feature = self.get_voxels(x, intrins)  # B x C x X x Y
        # print(bev_feature.shape)  # B x 64 x 200 x 200
        seg = self.bevencode(bev_feature)  # B x 1 x X x Y
        
        return bev_feature[:, :, :196, :], seg[:, :, :196, :]


if __name__ == '__main__':
    with open('/home/zjy/workspace/semi-BEVseg/BEVmix/configs/config.yml') as f:
        config = CfgNode.load_cfg(f)
    config.img_size = [600, 800]
    config.num_class = 14

    model = model_by_2dTo3d(config).cuda()
    total_num = sum(p.numel() for p in model.parameters())
    print('total_num =', total_num)

    img = torch.rand(1, 3, 600, 800).cuda()  # shape of input = 24 x 3 x 224 x 480
    k = torch.tensor([[[200., 0, 400], [0, 200, 300], [0, 0, 1]]]).cuda()

    bev_feature, seg = model(img, k)

    print(bev_feature.shape, seg.shape)

