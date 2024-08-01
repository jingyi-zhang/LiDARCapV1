import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.geometry import rot6d_to_rotmat
from st_gcn import STGCN
from pointnet2 import PointNet2Encoder, MqhPointNet2Encoder, PointNet2Regressor, PointNet2RegressorLight


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(n_hidden, n_hidden, n_rnn_layer,
                          batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(n_input, n_hidden)

        self.linear2 = nn.Linear(n_hidden * 2, n_output)

    def forward(self, x):  # (B, T, D)
        x = self.rnn(F.relu(F.dropout(self.linear1(x)), inplace=True))[0]       # 此处F.dropout用法有误，但估计不影响性能
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
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.pose_s1 = RNN(1024, 24 * 3, 1024)

        self.pose_s2 = STGCN(3 + 1024)

    def forward(self, data):

        pred = {}

        x = self.encoder(data)  # (B, T, D)
        B, T, _ = x.shape
        full_joints = self.pose_s1(x)  # (B, T, 24, 3)

        rot6ds = self.pose_s2(torch.cat((full_joints.reshape(
            B, T, 24, 3), x.unsqueeze(-2).repeat(1, 1, 24, 1)), dim=-1))
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T, D)
        rotmats = rot6d_to_rotmat(
            rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
        pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)
        pred = {**data, **pred}
        return pred

class LabelRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MqhPointNet2Encoder()
        self.pose_s1 = RNN(1024, 24 * 3, 1024)

        self.pose_s2 = STGCN(3 + 1024)

    def forward(self, data):

        pred = {}

        x = self.encoder(data)  # (B, T, D)
        B, T, _ = x.shape
        full_joints = self.pose_s1(x)  # (B, T, 24, 3)

        rot6ds = self.pose_s2(torch.cat((full_joints.reshape(
            B, T, 24, 3), x.unsqueeze(-2).repeat(1, 1, 24, 1)), dim=-1))
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T, D)
        rotmats = rot6d_to_rotmat(
            rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
        pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)
        pred = {**data, **pred}
        return pred


class SegmentRegressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if 'PointNet2RegressorLight' in cfg and cfg.PointNet2RegressorLight:
            self.encoder = PointNet2RegressorLight(cfg=cfg)
        else:
            self.encoder = PointNet2Regressor(feature_dim=0, cfg=cfg)

        self.pose_s1 = RNN(1024, 24 * 3, 1024)

        self.pose_s2 = STGCN(3 + 1024)

        if 'proj_joint_feature' in cfg and cfg.proj_joint_feature:
            self.proj = nn.Linear(1024, 24 * 1027, bias=False)

        if 'dropout_joint_feature' in cfg and cfg.dropout_joint_feature:
            self.joint_feature_token = nn.Parameter(torch.zeros(1, 1, 24, 1027))
            self.mask_joint_n = 12
            self.mask_frame_n = 8

        self.cfg = cfg

    def forward(self, data):

        pred = {}

        segment_pred, x = self.encoder(data)  # (B, T, D)
        B, T, _ = x.shape
        full_joints = self.pose_s1(x)  # (B, T, 24, 3)

        if self.cfg.proj_joint_feature:
            joints_feautre = self.proj(x).reshape(B, T, 24, 1027)
            if 'dropout_joint_feature' in self.cfg and self.cfg.dropout_joint_feature:
                mask_frame_n, mask_joint_n = np.random.randint(self.mask_frame_n), np.random.randint(self.mask_joint_n)
                if mask_frame_n and mask_joint_n:
                    dropout_joints_idx = torch.from_numpy(
                        np.random.choice(np.arange(24), size=mask_joint_n, replace=False).reshape(1, 1, -1))
                    joints_feautre[torch.arange(B).reshape(-1, 1, 1), np.random.choice(np.arange(T), size=mask_frame_n, replace=False).reshape(1, -1, 1), dropout_joints_idx] = \
                        self.joint_feature_token[torch.zeros((B, 1, 1), dtype=torch.long), torch.zeros(1, mask_frame_n, 1, dtype=torch.long), dropout_joints_idx]
                    #joints_feautre[torch.arange(B).reshape(-1, 1, 1), np.random.choice(np.arange(T), size=mask_frame_n, replace=False).reshape(1, -1, 1), dropout_joints_idx] = 0
            rot6ds = self.pose_s2(joints_feautre)
        else:
            rot6ds = self.pose_s2(torch.cat((full_joints.reshape(
                B, T, 24, 3), x.unsqueeze(-2).repeat(1, 1, 24, 1)), dim=-1))

        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T, D)
        rotmats = rot6d_to_rotmat(
            rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
        pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)
        pred['pred_segment']=segment_pred
        pred = {**data, **pred}
        return pred



if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    #model = VotingRegressor().cuda()
    from dataloader import Lidarcapv2_Dataset
    dataset = Lidarcapv2_Dataset(dataset_ids=[51804, 51805, 51807])
    #dataset = Lidarcapv2_Dataset(cfg.TrainDataset)
    loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=4)

    model = LabelRegressor().cuda()
    data = {k: v.cuda() for k, v in next(iter(loader)).items()}
    data['human_points'] = torch.cat((data['human_points'], data['body_label'].float().unsqueeze(-1)), dim=-1)
    model(data)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    model = Regressor().cuda()
    x = torch.randn((4, 16, 512, 3)).cuda()
    x = {'human_points': x}
    # Warn-up
    import time
    for _ in range(5):
        start = time.time()
        outputs = model(x)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        outputs = model(x)
    print(prof.table())

