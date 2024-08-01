# -*- coding: utf-8 -*-
# @Author  : Zhang.Jingyi

"""
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule\
return predict subsampled mesh and camera parameters 
"""

from __future__ import division

import torch
import torch.nn as nn

from .graph_layers import GraphResBlock, GraphLinear
from .resnet3d import ResNet34, ResNet18, ResNet50


class GraphCNN(nn.Module):
    # A: mesh.adjmat---->the graph adjacnecy matrix at the specified subsampling level;
    # ref_vertices: mesh.ref_vertices---->the template vertices at the specified subsampling level;
    # num_layers
    # num_channels

    # def __init__(self, A, ref_vertices, num_layers=5, num_channels=512):
    def __init__(self, A, ref_vertices):
        super(GraphCNN, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices
        self.resnet = ResNet18(3, 512, 3)
        layers = [GraphLinear(3 + 512, 2 * 256)]
        layers.append(GraphResBlock(2 * 256, 256, A))
        for i in range(5):
            layers.append(GraphResBlock(256, 256, A))
        self.shape = nn.Sequential(GraphResBlock(256, 64, A),
                                   GraphResBlock(64, 32, A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))
        self.gc = nn.Sequential(*layers)
        self.camera_fc = nn.Sequential(nn.GroupNorm(256 // 8, 256),
                                       nn.ReLU(inplace=True),
                                       GraphLinear(256, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(A.shape[0], 3))

    def forward(self, sparse_pc, batch_size):
        """Forward pass
        Inputs:
            pointcloud: size = (B, 3, N)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        ref_vertices = self.ref_vertices[None, :, :].expand(
            batch_size, -1, -1)  # (B, 1723, 3)
        resnet_output = self.resnet(sparse_pc)
        resnet_feature = resnet_output.F  # shape size: (B, 512)
        image_enc = resnet_feature.view(
            batch_size, 512, 1).expand(-1, -1, ref_vertices.shape[-1])  # (B, 512, 1723)
        x = torch.cat([ref_vertices, image_enc], dim=1)  # (B, 515, 1723)
        x = self.gc(x)  # (B, 256, 1723)
        shape = self.shape(x)  # (B, 3, 1723)
        #camera = self.camera_fc(x).view(batch_size, 3)
        # return shape, camera
        return shape
