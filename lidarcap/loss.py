import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import SMPL
from modules.geometry import axis_angle_to_rotation_matrix
from utils.geometric_layers import rodrigues


def batch_pc_normalize(pc):
    pc -= pc.mean(1, True)
    return pc / pc.norm(dim=-1, keepdim=True).max(1, True)[0]



class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion_param = nn.MSELoss()
        self.criterion_joints = nn.MSELoss()
        self.criterion_vertices = nn.MSELoss()
        # self.chamfer_loss = ChamferLoss()
        self.smpl = SMPL().cuda()

    def forward(self, **kw):
        B, T = kw['human_points'].shape[:2]
        gt_pose = kw['pose']
        gt_rotmats = axis_angle_to_rotation_matrix(
            gt_pose.reshape(-1, 3)).reshape(B, T, 24, 3, 3)

        #@mqh
        #gt_full_joints = kw['full_joints'].reshape(B, T, 24, 3)
        gt_full_joints = self.smpl.get_full_joints(self.smpl(gt_rotmats.reshape(-1, 24, 3, 3), gt_rotmats.new_zeros((B * T, 10)))).reshape(B, T, 24, 3) if 'full_joints' not in kw else kw['full_joints']
        gt_full_joints = gt_full_joints.detach()

        details = {}

        if 'pred_rotmats' in kw:
            pred_rotmats = kw['pred_rotmats'].reshape(B, T, 24, 3, 3)
            loss_param = self.criterion_param(pred_rotmats, gt_rotmats)
            details['loss_param'] = loss_param

            pred_human_vertices = self.smpl(
                pred_rotmats.reshape(-1, 24, 3, 3), pred_rotmats.new_zeros((B * T, 10)))
            pred_smpl_joints = self.smpl.get_full_joints(
                pred_human_vertices).reshape(B, T, 24, 3)
            loss_smpl_joints = self.criterion_joints(
                pred_smpl_joints, gt_full_joints)
            details['loss_smpl_joints'] = loss_smpl_joints

            # gt_human_vertices = self.smpl(
            #     gt_rotmats.reshape(-1, 24, 3, 3), torch.zeros((B * T, 10)).cuda())
            # loss_vertices = self.criterion_vertices(
            #     pred_human_vertices, gt_human_vertices)
            # details['loss_vertices'] = loss_vertices

        if 'pred_full_joints' in kw:
            pred_full_joints = kw['pred_full_joints']
            loss_full_joints = self.criterion_joints(
                pred_full_joints, gt_full_joints)
            details['loss_full_joints'] = loss_full_joints
        # human_points = kw['human_points'].reshape(BT, -1, 3)
        # loss_shape = chamfer_distance(human_points, kw['pred_vertices'])[0]
        # loss_shape.requires_grad_()

        # gt_human_vertex = kw['human_vertex']

        loss = 0
        for _, v in details.items():
            loss += v
        details['loss'] = loss
        return loss, details

class mqh_Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.criterion_param = nn.MSELoss()
        self.criterion_joints = nn.MSELoss()
        self.criterion_vertices = nn.MSELoss()
        # self.chamfer_loss = ChamferLoss()
        self.criterion_segment = F.nll_loss
        self.smpl = SMPL().cuda()
        self.cfg = cfg

    def forward(self, **kw):
        B, T, N, C = kw['human_points'].shape

        assert 'rotmats' in kw and 'full_joints' in kw, 'Must use dataset which have gt_rotmats and full_joints in order to reduce computational cost!'

        gt_rotmats = axis_angle_to_rotation_matrix(kw['pose'].reshape(-1, 3)).reshape(B, T, 24, 3, 3) if 'rotmats' not in kw else kw['rotmats']
        gt_full_joints = self.smpl.get_full_joints(self.smpl(gt_rotmats.reshape(-1, 24, 3, 3), torch.zeros((B * T, 10)).cuda())).reshape(B, T, 24, 3) if 'full_joints' not in kw else kw['full_joints']

        gt_rotmats = gt_rotmats.detach()    # 保险起见
        gt_full_joints = gt_full_joints.detach()

        loss_dict = {}
        others = {}
        loss = 0

        if 'pred_rotmats' in kw:
            pred_rotmats = kw['pred_rotmats'].reshape(B, T, 24, 3, 3)
            loss_param = self.criterion_param(pred_rotmats, gt_rotmats)
            loss_dict['loss_param'] = loss_param

            pred_human_vertices = self.smpl(
                pred_rotmats.reshape(-1, 24, 3, 3), pred_rotmats.new_zeros((B * T, 10)))
            pred_smpl_joints = self.smpl.get_full_joints(
                pred_human_vertices).reshape(B, T, 24, 3)
            loss_smpl_joints = self.criterion_joints(
                pred_smpl_joints, gt_full_joints)
            loss_dict['loss_smpl_joints'] = loss_smpl_joints

            loss += loss_param + loss_smpl_joints

            others['pred_smpl_joints'] = pred_smpl_joints

        if 'pred_full_joints' in kw and self.cfg.use_joints_loss:
            pred_full_joints = kw['pred_full_joints']
            loss_full_joints = self.criterion_joints(
                pred_full_joints, gt_full_joints)
            loss_dict['loss_full_joints'] = loss_full_joints
            loss += loss_full_joints

        if 'pred_segment' in kw and self.cfg.use_seg_loss:
            pred_segment = kw['pred_segment'].contiguous().view(-1, 2)

            target_segment = kw['body_label'].long().contiguous().view(-1)
            #target_segment = torch.zeros((*kw['human_points'].shape[:-1], 1)).long().contiguous().view(-1).cuda(kw['human_points'].device)

            loss_segment = self.criterion_segment(pred_segment, target_segment, weight=torch.tensor([1, 1]).float().cuda())
            loss_dict['loss_segment'] = loss_segment

            loss += loss_segment

        loss_dict['loss'] = loss
        return loss_dict, others

class GCNLoss(nn.Module):
    """ ReLoss: Reinforce Learning Loss
    """

    def __init__(self):
        super(GCNLoss, self).__init__()

        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_shape = nn.L1Loss()
        self.criterion_regr = nn.MSELoss()
        self.smpl = SMPL()

    # noinspection PyUnresolvedReferences
    def forward(self, **kw):
        '''
        Args:
            **kw: dict('img_orig', 'img', 'imgname', 'pose_3d', 'human_vertex',
                        'human_points', 'timestamp', 'pose', 'beta', 'keypoints',
                        'has_smpl', 'has_pose_3d', 'has_vertex', 'has_human_points',
                        'has_timestamp', 'scale', 'center', 'maskname', 'partname',
                        'pred_vertices', 'pred_keypoints_3d', 'pred_keypoints_3d_smpl',
                        'pred_vertices_smpl', 'pred_rotmat', 'pred_shape')
            pose_3d:non
            human_vertex: 3D human vertex
            human_points: 3D human points
            pose: smpl parameters
            beta: smpl parameters
            keypoints: non
            has_smpl: 1
            has_pose_3d:0
            has_vertex:1
            has_human_points:1
            has_timestamp:1
            scale:
            center:
            maskname:
            partname:
            pred_vertices:
            pred_keypoints_3d: no_GT
            pred_keypoints_3d_smpl: no_GT
            pred_vertices_smpl:
            pred_rotmat:
            pred_shape:


        Return: Loss, details
        '''
        # get the prediction of network
        loss_part = {}
        # get the ground truth of network
        gt_smpl_shape = kw.get('betas')
        B, T = gt_smpl_shape.shape[:2]
        gt_smpl_shape = gt_smpl_shape.reshape(B * T, -1)
        gt_smpl_pose = kw.get('pose').reshape(B * T, -1)  # 8,72

        gt_human_vertex = kw.get('human_vertex').reshape(B * T, -1, 3)
        gt_keypoints = self.smpl.get_full_joints(gt_human_vertex)
        batch_size = gt_smpl_pose.shape[0]
        pose_cube = gt_smpl_pose.view(-1, 3)
        R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
        R = R.view(batch_size, 24, 3, 3)

        if kw.get('pred_shape') is not None:
            pred_smpl_shape = kw.get('pred_shape')
            loss_smpl_shape = self.criterion_shape(
                pred_smpl_shape, gt_smpl_shape)
            out = dict(loss_smpl_shape=loss_smpl_shape)
            loss_part.update(out)

        if kw.get('pred_rotmat') is not None:
            pred_smpl_rotmat = kw.get('pred_rotmat')
            loss_smpl_rotmat = self.criterion_shape(pred_smpl_rotmat, R)
            out = dict(loss_smpl_rotmat=loss_smpl_rotmat)
            loss_part.update(out)

        if kw.get('pred_vertices') is not None:
            pred_vertices = kw.get('pred_vertices')
            loss_shape = self.shape_loss(
                pred_vertices, gt_human_vertex)
            out = dict(loss_shape=loss_shape)
            loss_part.update(out)

        if kw.get('pred_vertices_smpl') is not None:
            pred_vertices_smpl = kw.get('pred_vertices_smpl')
            loss_shape_smpl = self.shape_loss(
                pred_vertices_smpl, gt_human_vertex)
            out = dict(loss_shape_smpl=loss_shape_smpl)
            loss_part.update(out)

        if kw.get('pred_keypoints') is not None:
            pred_keypoints = kw.get('pred_keypoints')
            loss_keypoints = self.criterion_keypoints(
                gt_keypoints, pred_keypoints).mean()
            out = dict(loss_keypoints=loss_keypoints)
            loss_part.update(out)

        loss = 0
        details = loss_part
        for k, v in loss_part.items():
            loss += v
        details.update(dict(loss=loss))

        return loss, details
    #

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
        """Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d,
                                                gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence
        """
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] +
                         gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (
                pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        return self.criterion_shape(pred_vertices, gt_vertices)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        """Compute SMPL parameter loss for the examples that SMPL annotations are available."""
        pred_rotmat_valid = pred_rotmat[has_smpl == 1].view(-1, 3, 3)
        gt_rotmat_valid = rodrigues(gt_pose[has_smpl == 1].view(-1, 3))
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(
                pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(
                pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas
