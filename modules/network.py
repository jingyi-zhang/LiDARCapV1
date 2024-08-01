import torch

from ._init_path import *
from .graph_cnn import GraphCNN
from .smpl_param_regressor import SMPLParamRegressor
from .joints_location_regressor import Joints_Loc_Regressor
from .mesh import Mesh
from .smpl import SMPL
import MinkowskiEngine as ME


class Basic_Mesh_NET(torch.nn.Module):
    def __init__(self, train_step):
        super().__init__()

        self.mesh = Mesh()
        self.smpl = SMPL()
        self.train_step = train_step

        self.graph_cnn = GraphCNN(self.mesh.adjmat, self.mesh.ref_vertices.t())
        self.smpl_param_regressor = SMPLParamRegressor()
        self.joints_loc_regressor = Joints_Loc_Regressor()

    def forward(self, data):
        coordinate = []
        human_points = data['human_points']
        points_num = data['points_num']
        N = human_points.size(-2)
        human_points = human_points.reshape(-1, N, 3)
        points_num = points_num.reshape(-1, )
        batch_size = human_points.shape[0]

        for sample_ind in range(batch_size):
            cur_points_num = points_num[sample_ind]
            cur_human_points = human_points[sample_ind][:cur_points_num, :]
            coordinate.append(cur_human_points / 0.02)
        coordinates = ME.utils.batched_coordinates(
            coordinate, dtype=torch.float32)
        features = torch.vstack(coordinate)
        inputs = ME.SparseTensor(features, coordinates, device='cuda')

        pred = {}

        if self.train_step == 'first':
            with torch.set_grad_enabled(True):
                self.graph_cnn.train()
                self.joints_loc_regressor.train()

                pred_vertices_sub = self.graph_cnn(
                    inputs, batch_size)  # (B, 1723, 3)
                pred_vertices = self.mesh.upsample(
                    pred_vertices_sub.transpose(1, 2))  # (B, 6890, 3)
                pred_keypoints = self.joints_loc_regressor(pred_vertices)

                out = dict(pred_vertices_sub=pred_vertices_sub,
                           pred_vertices=pred_vertices,
                           pred_keypoints=pred_keypoints)
                pred = {**pred, **{k: v for k, v in out.items()}}

            pred = {**data, **pred}

            return pred

        elif self.train_step == 'second':
            with torch.set_grad_enabled(False):
                self.graph_cnn.eval()
                self.joints_loc_regressor.eval()
                pred_vertices_sub = self.graph_cnn(
                    inputs, batch_size)  # (B, 1723, 3)
                pred_vertices = self.mesh.upsample(
                    pred_vertices_sub.transpose(1, 2))  # (B, 6890,3)
                pred_keypoints = self.joints_loc_regressor(pred_vertices)

            with torch.set_grad_enabled(True):
                self.smpl_param_regressor.train()
                x = pred_vertices_sub.transpose(1, 2).detach()  # (B, 1723, 3)
                x = torch.cat(
                    [x, self.mesh.ref_vertices[None, :,
                                               :].expand(batch_size, -1, -1)],
                    dim=-1)  # (B, 1723, 6)
                pred_rotmat = self.smpl_param_regressor(x)
                B = pred_rotmat.size(0)
                pred_vertices_smpl = self.smpl(
                    pred_rotmat, data['betas'].reshape(B, -1))

                out = dict(pred_keypoints=pred_keypoints,
                           pred_vertices=pred_vertices,
                           pred_vertices_smpl=pred_vertices_smpl,
                           pred_rotmat=pred_rotmat,
                           )
                pred = {**pred, **{k: v for k, v in out.items()}}

            pred = {**data, **pred}

            return pred

        else:
            with torch.set_grad_enabled(True):
                self.graph_cnn.train()
                self.joints_loc_regressor.train()
                pred_vertices_sub = self.graph_cnn(
                    inputs, batch_size)  # (B, 1723, 3)
                pred_vertices = self.mesh.upsample(
                    pred_vertices_sub.transpose(1, 2))  # (B, 6890,3)
                pred_keypoints = self.joints_loc_regressor(pred_vertices)

            with torch.set_grad_enabled(True):
                self.smpl_param_regressor.train()
                x = pred_vertices_sub.transpose(1, 2).detach()  # (B, 1723, 3)
                x = torch.cat(
                    [x, self.mesh.ref_vertices[None, :,
                                               :].expand(batch_size, -1, -1)],
                    dim=-1)  # (B, 1723, 6)
                pred_rotmat = self.smpl_param_regressor(x)
                B = pred_rotmat.size(0)
                pred_vertices_smpl = self.smpl(
                    pred_rotmat, data['betas'].reshape(B, -1))

                out = dict(pred_keypoints=pred_keypoints,
                           pred_vertices=pred_vertices,
                           pred_vertices_smpl=pred_vertices_smpl,
                           pred_rotmat=pred_rotmat,
                           )
                pred = {**pred, **{k: v for k, v in out.items()}}

            pred = {**data, **pred}

            return pred
