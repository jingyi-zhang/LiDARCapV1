import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.smpl import SMPL, get_smpl_vertices
from modules.geometry import rot6d_to_rotmat

from modules.pointlstm import PointLSTMEncoder, MLPBlock
from modules.p4transformer import P4TransformerExtractor

# Sparse Point Pose Regression


class SPPR(nn.Module):
    def __init__(self, extractor='pointlstm', regress_mode='hierarchy', train_step='first'):
        super().__init__()
        self.extractor = PointLSTMEncoder(
        ) if extractor == 'pointlstm' else P4TransformerExtractor()
        self.attn_conv = MLPBlock([1024, 256, 24], 2)
        # self.head
        self.head_full_joints = nn.Sequential(
            MLPBlock([1024, 256, 64], 2),
            MLPBlock([64, 16, 3], 2, False)
        )

        self.train_step = train_step

        self.head_full = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 24 * 3)
        )

    def forward(self, data):
        x = self.extractor(data)  # (B, 512, T, 64)
        B, _, T, _ = x.shape
        # N2 = x.size(-1)
        # attn_weights = F.softmax(self.attn_conv(x), -1)  # (B, 24, T, 64)
        # x = torch.bmm(
        #     x.transpose(1, 2).reshape(B * T, -1, N2),
        #     attn_weights.permute(0, 2, 3, 1).reshape(B * T, N2, -1)
        # ).reshape(B, T, 1024, 24).transpose(1, 2)

        x = F.adaptive_avg_pool2d(x, (T, 24))
        full_joints = self.head_full_joints(x)

        # x = self.extractor(data)
        # B, T, _ = x.shape
        # full_joints = self.head_full(x)

        pred = {}
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)
        pred = {**data, **pred}
        return pred


if __name__ == '__main__':
    model = SPPR().cuda()
    x = torch.randn((2, 16, 512, 3)).cuda()
    print(model(x).shape)
