import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.geometry import rot6d_to_rotmat
from modules.hmr import HMR as hmr
from modules.st_gcn import STGCN
from modules.p4transformer import P4TransformerEncoder, P4TransformerExtractor, P4TransformerVoter
from modules.pointnet2 import PointNet2Regressor, PointNet2Encoder
from modules.pointlstm import PointLSTM, PointLSTMEncoder, MLPBlock


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(n_hidden, n_hidden, n_rnn_layer,
                          batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(n_input, n_hidden)

        self.linear2 = nn.Linear(n_hidden * 2, n_output)

    def forward(self, x):  # (B, T, D)
        x = self.rnn(F.relu(F.dropout(self.linear1(x)), inplace=True))[0]
        return self.linear2(x)


class NaiveRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.temporal_encoder = RNN(1024, 24 * 6, 1024)

    def forward(self, data):
        pred = {}
        x = self.encoder(data)  # (B, T, D)
        B, T, _ = x.shape
        rot6ds = self.temporal_encoder(x)  # (B, T, 24 * 6)
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T * 24, 6)
        rotmats = rot6d_to_rotmat(
            rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
        pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)
        pred = {**data, **pred}
        return pred


class Regressor(nn.Module):
    def __init__(self, train_step='first'):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.pose_s1 = RNN(1024, 24 * 3, 1024)

        self.train_step = train_step

        # self.encoder = P4TransformerEncoder()
        # self.pose_s1 = nn.Sequential(
        #     nn.LayerNorm(1024),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 24 * 3)
        # )

        if train_step == 'second':
            # self.pose_s2 = RNN(24 * 3 + 1024, 24 * 6, 1024)
            self.pose_s2 = STGCN(3 + 1024)

    def forward(self, data):

        pred = {}

        x = self.encoder(data)  # (B, T, D)
        B, T, _ = x.shape
        full_joints = self.pose_s1(x)  # (B, T, 24, 3)

        if self.train_step == 'second':
            # rot6ds = self.pose_s2(torch.cat((full_joints, x), dim=-1))
            rot6ds = self.pose_s2(torch.cat((full_joints.reshape(
                B, T, 24, 3), x.unsqueeze(-2).repeat(1, 1, 24, 1)), dim=-1))
            rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T, D)
            rotmats = rot6d_to_rotmat(
                rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
            pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)
        pred = {**data, **pred}
        return pred


class AttentionRegressor(nn.Module):
    def __init__(self, train_step='second'):
        super().__init__()
        self.extractor = P4TransformerExtractor()
        self.attn_conv = MLPBlock([1024, 256, 24], 2)

        self.head_full_joints = nn.Sequential(
            MLPBlock([1024, 256, 64], 2),
            MLPBlock([64, 16, 3], 2, False)
        )

        if train_step == 'second':
            # self.pose_s2 = RNN(24 * 3 + 1024, 24 * 6, 1024)
            self.pose_s2 = STGCN(3)
        self.train_step = train_step

    def forward(self, data):
        x = self.extractor(data)  # (B, 512, T, 64)
        B, _, T, _ = x.shape
        N2 = x.size(-1)
        attn_weights = F.softmax(self.attn_conv(
            x) + 1e-8, -1)  # (B, 24, T, 64)
        x = torch.bmm(
            x.transpose(1, 2).reshape(B * T, -1, N2),
            attn_weights.permute(0, 2, 3, 1).reshape(B * T, N2, -1)
        ).reshape(B, T, 1024, 24).transpose(1, 2)
        full_joints = self.head_full_joints(x)  # (B, 3, T, 24)
        full_joints = full_joints.permute(0, 2, 3, 1)

        pred = {}
        pred['pred_full_joints'] = full_joints
        if self.train_step == 'second':
            rot6ds = self.pose_s2(full_joints)
            rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T, D)
            rotmats = rot6d_to_rotmat(
                rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
            pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)

        pred = {**data, **pred}
        return pred


class VotingRegressor(nn.Module):
    def __init__(self, train_step='first'):
        super().__init__()
        self.voter = P4TransformerVoter()

        self.outconv = nn.Conv2d(
            in_channels=128, out_channels=96, kernel_size=1, stride=1, padding=0)

        if train_step == 'second':
            self.pose_s2 = STGCN(3)
        self.train_step = train_step

    def get_voted_joints(self, x):
        return torch.mean(x[..., :3] * F.softmax(x[..., 3:], 2), 2)

    def forward(self, data):
        x = data['human_points']  # (B, T, N, 3)
        pred = {}
        B, T, N = x.shape[:3]
        features = self.voter(x)  # (B, 128, T, N)
        x = self.outconv(features)  # (B, 96, T, N)
        x = x.permute(0, 2, 3, 1).reshape(B, T, N, 24, 4)
        full_joints = self.get_voted_joints(x)  # (B, T, 24, 3)
        pred['pred_full_joints'] = full_joints

        if self.train_step == 'second':
            rot6ds = self.pose_s2(full_joints)
            rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T, D)
            rotmats = rot6d_to_rotmat(
                rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
            pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)

        pred = {**data, **pred}

        return pred


# 相加的话 考虑加入attention
# 考虑加入残差连接


class FusionRegressor(nn.Module):
    def __init__(self, train_step='first'):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.train_step = train_step

        self.pose_s1 = RNN(1024, 24 * 3, 1024)
        self.pose_s2 = RNN(24 * 3 + 1024, 24 * 6, 1024)
        self.hmr = hmr(False)
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True))

    def forward(self, data):

        pred = {}

        B, T, C, H, W = data['depths'].shape

        x = self.fc1(self.encoder(data))  # (B, T, D)
        y = self.hmr.feature_extractor(data['depths'].reshape(B * T, C, H, W))
        y = self.fc2(y).reshape(B, T, -1)
        x = x + y
        # x = torch.cat((x, y), dim=-1)

        # x = self.hmr.feature_extractor(data['depths'].reshape(B * T, C, H, W))
        # x = self.fc2(x).reshape(B, T, -1)

        full_joints = self.pose_s1(x)
        if self.train_step == 'second':
            rot6ds = self.pose_s2(torch.cat((full_joints, x), dim=-1))
            rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T, D)
            rotmats = rot6d_to_rotmat(
                rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
            pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)
            # pred['pred_vertices'] = get_smpl_vertices(data['trans'].reshape(
            #     B * T, 3), rotmats.reshape(B * T, 24, 3, 3), data['betas'].reshape(B * T, 10), self.smpl)

        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)

        pred = {**data, **pred}
        return pred


if __name__ == '__main__':
    model = VotingRegressor().cuda()
    x = torch.randn((4, 16, 512, 3)).cuda()
    model(x)
