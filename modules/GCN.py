import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.resnet3d import ResNet18, ResNet101
import torch
import torch.nn as nn
import MinkowskiEngine as ME


class GCNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet101(3, 2048, 3)
        

    def forward(self, data):
        B, T, N, _ = data['human_points'].shape
        human_points = data['human_points'].reshape(-1, N, 3)
        points_num = data['points_num'].reshape(-1, 1)

        # 除以0.02 是要体素化，必须（jingyi）
        coordinate = [human_points[i][:points_num[i]] /
                      0.02 for i in range(B * T)]

        coordinates = ME.utils.batched_coordinates(
            coordinate, dtype=torch.float32)
        features = torch.vstack(coordinate)
        x = ME.SparseTensor(features, coordinates, device='cuda')
        x = self.resnet(x).F
        x = x.reshape(B, T, -1)
        return x
