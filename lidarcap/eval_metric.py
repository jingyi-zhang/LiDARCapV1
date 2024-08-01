import sys
import time

import h5py
import numpy as np
from dataloader import Lidarcapv2_Dataset

from metric_tool.compare_vibe_hmr import mqh_output_metric

import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import socketio
client = socketio.Client()


def load_model(ckpt_path, type_name='Regressor'):
    sys.path.insert(0, os.path.dirname(ckpt_path))     # 避免regressory引入lidarcap.pointnet2 和 lidarcap.stg_gcn
    import regressor
    import importlib
    importlib.reload(regressor)
    assert os.path.dirname(regressor.__file__) == os.path.dirname(ckpt_path)

    model = regressor.__dict__[type_name]().cuda()

    if ckpt_path is not None:
        print('use ckpt_path', ckpt_path)
        saved_model = torch.load(ckpt_path, map_location='cuda:0')
        saved_state_dict = saved_model['state_dict'] if 'state_dict' in saved_model else saved_model['gen_state_dict']
        model_dict = model.state_dict()
        if model_dict.keys() != saved_state_dict.keys():
            print('[WARN]:The given ckpt_path is not equal to model!')
        if next(iter(saved_state_dict.keys())).startswith('module.'):
            model.load_state_dict({k.replace('module.', ''): v for k, v in saved_state_dict.items()})
        else:
            model_dict.update({k: v for k, v in saved_state_dict.items() if k in model_dict})
            model.load_state_dict(model_dict)
    else:
        print('use raw model!')

    sys.path.remove(os.path.dirname(ckpt_path))

    return model



def eval_lidarhuman26m(model_and_loss, ids, bs, ip, tag, wseg):
    from modules import SMPL
    smpl = SMPL().cuda()

    testset = Lidarcapv2_Dataset(dataset_ids=ids, ret_raw_pc=True)
    valid_loader = torch.utils.data.DataLoader(testset, num_workers=0, batch_size=bs, shuffle=False)

    model= model_and_loss

    #loss_func = mqh_loss()

    all_pred_joints = []
    all_gt_joints = []

    for index, data in enumerate(tqdm(valid_loader, ncols=60)):
        model.eval()
        with torch.no_grad():
            inputs = {k: v.cuda() for k, v in data.items()}

            #pc = inputs['human_points']
            #pc[pc[:, :, :, 2] < pc.mean(dim=-2)[:, :, 2].unsqueeze(-1) - 0.1] = 0

            output = model(inputs)

            #loss_dict, others = loss_func(**output)

            B, T = output['human_points'].shape[:2]
            pred_human_vertices = smpl(output['pred_rotmats'].reshape(-1, 24, 3, 3),
                                       output['pred_rotmats'].new_zeros((B * T, 10)))
            pred_smpl_joints = smpl.get_full_joints(pred_human_vertices).reshape(B, T, 24, 3)


            pred_joints = pred_smpl_joints.reshape(-1, 24, 3).cpu().numpy()
            gt_joints = output['full_joints'].reshape(-1, 24, 3).cpu().numpy()
            all_pred_joints.append(pred_joints)
            all_gt_joints.append(gt_joints)

            pred_poses = (R.from_matrix(output['pred_rotmats'].cpu().reshape(-1, 3, 3))).as_rotvec().reshape(
                *output['pred_rotmats'].shape[:2], 72)

            batch_indexes = output['index'].tolist()

            for bi in batch_indexes:
                if wseg:
                    pred_segment = output['pred_segment'][bi % bs].argmax(dim=-1)
                    gt_segment = output['body_label'][bi % bs]

                pred_p = pred_poses[bi % bs]
                gt_p = data['pose'][bi % bs].numpy()
                pc = data['point_clouds'][bi % bs].numpy()
                trans = data['trans'][bi % bs].numpy()

                """
                for seqi in range(16):
                    offset = bi * 16
                    pc_data = {seqi + offset: {'pc': pc[seqi].tolist()}}
                    pred_pose_data = {seqi + offset: {'pose': pred_p[seqi].tolist(), 'trans': trans[seqi].tolist(), 'shader': 'defaultLit'}}
                    gt_pose_data = {seqi + offset: {'pose': gt_p[seqi].tolist(), 'trans': (trans[seqi] + np.array([0.5, 0.5, 0])).tolist(), 'shader': 'normals'}}
                    msg = {tag: {'objects': {'human_points': pc_data, 'pred_pose': pred_pose_data, 'gt_pose': gt_pose_data}}}
                    client.emit('update_seqs', data=msg)"""

                offset = bi * 16

                pc_data = {seqi + offset: {'pc': (pc[seqi] + np.array([0.5, 0.5, 0])).tolist()} for seqi in range(16)}
                if wseg:
                    pc_data = {seqi + offset: {'pc': (pc[seqi] + np.array([0.5, 0.5, 0])).tolist(), 'colors': pred_segment[seqi].tolist()} for seqi in range(16)}
                    #pc_data2 = {seqi + offset: {'pc': (pc[seqi] + np.array([0.5, 0.5, 0])).tolist(), 'colors': gt_segment[seqi].tolist()} for seqi in range(16)}

                gt_pose_data = {seqi + offset: {'pose': gt_p[seqi].tolist(),
                                                'trans': trans[seqi].tolist(),
                                                'shader': 'normals'} for seqi in range(16)}


                pred_pose_data = {seqi + offset: {'pose': pred_p[seqi].tolist(), 'trans': (trans[seqi] + np.array([0.5, 0.5, 0])).tolist(), 'shader': 'defaultLit'} for seqi in range(16)}

                mpjpe = mqh_output_metric(pred_joints.reshape(-1, 16, 24, 3)[bi % bs], gt_joints.reshape(-1, 16, 24, 3)[bi % bs])['mpjpe']
                label = {seqi + offset: {'pos': (pred_joints[(bi % bs) * 16 + seqi][15] + np.array([0.5, 0.5, 0.5] + trans[seqi])).tolist(), 'text': f"{tag}\nframe:{offset}\nmpjpe:{mpjpe:.1f}", 'color': [1., 1., 1., 1.]} for seqi in range(16)}


                msg = {tag: {'objects': {'human_points': pc_data, 'pred_pose': pred_pose_data, 'gt_pose': gt_pose_data, '3dlabel': label}}}

                if ip is not None:
                    client.emit('update_seqs', data=msg)


    all_pred_joints = np.concatenate(all_pred_joints, axis=0)
    all_gt_joints = np.concatenate(all_gt_joints, axis=0)

    for dataset_id, pred_joints, gt_joints in testset.split_list_by_dataset(all_pred_joints.reshape(-1, 24, 3), all_gt_joints.reshape(-1, 24, 3)):
        metric = mqh_output_metric(pred_joints.reshape(-1, 24, 3), gt_joints.reshape(-1, 24, 3))
        print(dataset_id, metric)

    return all_pred_joints, all_gt_joints


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=7)

    parser.add_argument('--bs', type=int, default=4)

    parser.add_argument('--ckpt_path', type=str, default='')

    parser.add_argument('--ip', type=str, default='127.0.0.1')

    parser.add_argument('--port', type=str, default=5671)

    parser.add_argument('--ids', nargs='+')

    parser.add_argument('--tag', type=str, default='default')

    args = parser.parse_args()

    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    try:
        print(f'try to connect :http://{args.ip}:{args.port}')
        client.connect(f'http://{args.ip}:{args.port}')
        print('conncet success!')
    except Exception as e:
        print(e)
        args.ip = None


    print('start load model:')

    r1 = eval_lidarhuman26m(
        load_model('output/apricot-galaxy-6/best_model.pth', 'SegmentRegressor'),
        args.ids, args.bs, args.ip, 'segv3', True)

    r2 = eval_lidarhuman26m(load_model('output/azure-dew-4/best_model.pth', 'Regressor'), args.ids, args.bs, args.ip, 'woseg', False)

    r1 = r1[0].reshape(-1, 16, 24, 3), r1[1].reshape(-1, 16, 24, 3)
    r2 = r2[0].reshape(-1, 16, 24, 3), r2[1].reshape(-1, 16, 24, 3)

    r1_mpjpe = np.array([mqh_output_metric(r1[0][i], r1[1][i])['mpjpe'] for i in range(len(r1[0]))])
    r2_mpjpe = np.array([mqh_output_metric(r2[0][i], r2[1][i])['mpjpe'] for i in range(len(r2[0]))])
    contrast = np.stack((r1_mpjpe, r2_mpjpe, np.arange(0, len(r1_mpjpe) * 16, 16) / 10), axis=1)
    print()
