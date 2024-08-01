# -*- coding: utf-8 -*-
# @Author  : Zhang.Jingyi
'''
Definition of regressor used for predicting 3D joints location
'''
import torch
import torch.nn as nn
from .layers import FCBlock, FCResBlock


class Joints_Loc_Regressor(nn.Module):
    def __init__(self):
        super(Joints_Loc_Regressor, self).__init__()

        self.layer = nn.Sequential(FCBlock(6890 * 3, 1024),
                                   FCResBlock(1024, 1024),
                                   FCResBlock(1024, 1024),
                                   nn.Linear(1024, 24 * 3))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.contiguous().view(batch_size, -1)
        x = self.layer(x)
        keypoints = x.view(-1, 24, 3).contiguous()

        return keypoints
