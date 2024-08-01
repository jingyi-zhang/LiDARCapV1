
from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule


class PointNet2Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=True
            )
        )

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data):
        x = data['human_points']  # (B, T, N, 3)
        B, T, N, _ = x.shape
        x = x.reshape(-1, N, _)  # (B * T, N, 3)
        xyz, features = self._break_up_pc(x)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        features = features.squeeze(-1).reshape(B, T, -1)
        return features

class MqhPointNet2Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[1, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=True
            )
        )

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data):
        x = data['human_points']  # (B, T, N, 3)
        B, T, N, _ = x.shape
        x = x.reshape(-1, N, _)  # (B * T, N, 3)
        xyz, features = self._break_up_pc(x)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        features = features.squeeze(-1).reshape(B, T, -1)
        return features


class PointNet2Regressor(nn.Module):
    def __init__(self, feature_dim, cfg=None):
        super().__init__()

        self.cfg = cfg

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[feature_dim, 32, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[512, 512, 512, 1024],
                use_xyz=True,
            )
        )

        self.SA_modules2 = nn.ModuleList()
        self.SA_modules2.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[257, 128, 128, 128],
                use_xyz=True,
            )
        )
        self.SA_modules2.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules2.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024],
                use_xyz=True,
            )
        )


        #self.SA_modules_mlp = PointnetSAModule(
            #mlp=[256, 256, 512, 1024], use_xyz=True)

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(
            mlp=[256 + feature_dim, 256, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 128, 512, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[1024 + 512, 512, 512]))

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 2, 1)

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data):
        r"""
            Forward pass of the network
            Parameters
            ----------
            data: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        # pointcloud = pointcloud.reshape(-1, *pointcloud.shape[-2:])
        pointcloud = data['human_points']  # (B, T, N, 4)
        B,T,N,_ = pointcloud.shape
        pointcloud = pointcloud.reshape(-1, N, _)
        xyz, features = self._break_up_pc(pointcloud[...,:3])
        features = None

        l_xyz, l_features = [xyz], [features]
        for i in range(0, len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # mid_xyz = l_xyz[2]
        # mid_feature = l_features[2]
        # human_feature = self.SA_modules_mlp(mid_xyz, mid_feature)
        # human_feature = human_feature[1].squeeze(-1).reshape(B, T, -1)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        x = self.drop1(F.relu(self.bn1(self.conv1(l_features[0]))))
        x = self.conv2(x)
        x = x.permute(0,2,1).contiguous()
        x = torch.softmax(x, dim=-1)
        x = x.view(B,T,N,-1)  # (B, T, N, 2)

        # noinspection PyUnresolvedReferences
        mask = (x[...,0] < x[...,1]).view(B*T, N)  # (B*T, N)
        # M = mask.sum(dim=1).max()  # int
        M = 512
        # h_xyz = xyz.new_zeros(B*T, M)  # (B*T, M)
        # for i, m in enumerate(mask):
        #     # m: (N,)
        #     h_xyz[i, :m.sum()] = xyz[i][m]
        #h_xyz = torch.stack([torch.cat((pts[m], pts.new_zeros(M-m.sum(), 3)), dim=0) for pts,m in zip(xyz,mask)])  # (B*T, M, 3)

        if 'no_mask_feature' in self.cfg and self.cfg.no_mask_feature:
            mask = torch.zeros((B*T, N)).to(l_features[0])

        h_xyz = xyz
        #h_xyz = xyz * mask.unsqueeze(-1)

        h_features = torch.cat((mask.unsqueeze(1), l_features[0]), dim=1)

        if not self.cfg.use_seg_feature:
            h_features = torch.zeros_like(h_features)

        h_xyz, h_features = [h_xyz], [h_features]
        for i in range(len(self.SA_modules2)):
            hi_xyz, hi_features = self.SA_modules2[i](h_xyz[i], h_features[i])
            h_xyz.append(hi_xyz)
            h_features.append(hi_features)

        human_feature = h_features[-1].squeeze(-1).reshape(B, T, -1)

        return x, human_feature

# 轻量级版本的PointNet2Regressor
class PointNet2RegressorLight(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        self.cfg = cfg

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[0, 32, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[512, 512, 512, 1024],
                use_xyz=True,
            )
        )

        self.SA_modules2 = nn.ModuleList()
        self.SA_modules2.append(
            PointnetSAModule(
                mlp=[1024, 1024, 1024, 1024],
                use_xyz=True,
            )
        )

        self.SA_modules3 = nn.ModuleList()
        self.SA_modules3.append(
            PointnetSAModule(
                mlp=[257, 256, 512, 1024],
                use_xyz=True,
            )
        )

        self.proj_global_feature = nn.Linear(2048, 1024, bias=False)


        #self.SA_modules_mlp = PointnetSAModule(
            #mlp=[256, 256, 512, 1024], use_xyz=True)

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(
            mlp=[256, 256, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 128, 512, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[1024 + 512, 512, 512]))

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 2, 1)

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data):
        r"""
            Forward pass of the network
            Parameters
            ----------
            data: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        # pointcloud = pointcloud.reshape(-1, *pointcloud.shape[-2:])
        pointcloud = data['human_points']  # (B, T, N, 4)
        B,T,N,_ = pointcloud.shape
        pointcloud = pointcloud.reshape(-1, N, _)
        xyz, features = self._break_up_pc(pointcloud[...,:3])
        features = None

        l_xyz, l_features = [xyz], [features]
        for i in range(0, len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # mid_xyz = l_xyz[2]
        # mid_feature = l_features[2]
        # human_feature = self.SA_modules_mlp(mid_xyz, mid_feature)
        # human_feature = human_feature[1].squeeze(-1).reshape(B, T, -1)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        x = self.drop1(F.relu(self.bn1(self.conv1(l_features[0]))))
        x = self.conv2(x)
        x = x.permute(0,2,1).contiguous()
        x = torch.softmax(x, dim=-1)
        x = x.view(B,T,N,-1)  # (B, T, N, 2)

        # noinspection PyUnresolvedReferences
        mask = (x[...,0] < x[...,1]).view(B*T, N)  # (B*T, N)
        # M = mask.sum(dim=1).max()  # int
        M = 512
        # h_xyz = xyz.new_zeros(B*T, M)  # (B*T, M)
        # for i, m in enumerate(mask):
        #     # m: (N,)
        #     h_xyz[i, :m.sum()] = xyz[i][m]
        #h_xyz = torch.stack([torch.cat((pts[m], pts.new_zeros(M-m.sum(), 3)), dim=0) for pts,m in zip(xyz,mask)])  # (B*T, M, 3)

        if 'no_mask_feature' in self.cfg and self.cfg.no_mask_feature:
            mask = torch.zeros((B*T, N)).to(l_features[0])

        h_xyz = l_xyz[-1]
        #h_xyz = xyz * mask.unsqueeze(-1)

        #h_features = torch.cat((mask.unsqueeze(1), l_features[-1]), dim=1)
        h_features = l_features[-1]


        h_xyz, h_features = [h_xyz], [h_features]
        for i in range(len(self.SA_modules2)):
            hi_xyz, hi_features = self.SA_modules2[i](h_xyz[i], h_features[i])
            h_xyz.append(hi_xyz)
            h_features.append(hi_features)

        human_feature = h_features[-1].squeeze(-1).reshape(B, T, -1)


        h_xyz = xyz
        h_features = torch.cat((mask.unsqueeze(1), l_features[0]), dim=1)
        if not self.cfg.use_seg_feature:
            h_features = torch.zeros_like(h_features)

        h_xyz, h_features = [h_xyz], [h_features]
        for i in range(len(self.SA_modules3)):      # TODO: 把sa2, sa3 的xyz，mask去掉，看下效果是否会变差
            hi_xyz, hi_features = self.SA_modules3[i](h_xyz[i], h_features[i])
            h_xyz.append(hi_xyz)
            h_features.append(hi_features)

        #human_feature = human_feature + h_features[-1].squeeze(-1).reshape(B, T, -1)
        human_feature = torch.cat((human_feature, h_features[-1].squeeze(-1).reshape(B, T, -1)), dim=-1)
        human_feature = self.proj_global_feature(human_feature)

        return x, human_feature

if __name__ == '__main__':
    encoder = PointNet2Encoder().cuda()
    input_pointclouds = torch.rand((4, 512, 3)).cuda()
    output_pointclouds = encoder(input_pointclouds)
    print(output_pointclouds.shape)
    # mesh = Mesh()
    # print(mesh.ref_vertices.shape)
