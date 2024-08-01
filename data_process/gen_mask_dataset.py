
import sys, os
sys.path.append('/cwang/home/mqh/lidarcap_fusion')

from modules.geometry import axis_angle_to_rotation_matrix
import torch

from tqdm import tqdm
import argparse

import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--bs', type=int, required=True)

parser.add_argument('--in_path', type=str, required=True)

parser.add_argument('--out_path', type=str, required=True)

args = parser.parse_args()



os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


def gen_mask_dataset(dataset_name, output_dataset_name, option):
    import h5py
    import random
    dataset = h5py.File(dataset_name, 'r')
    gt_pose = dataset['pose']
    B, T, _ = gt_pose.shape
    bs = args.bs

    from modules.smpl import SMPL
    smpl = SMPL().cuda()

    masked_point_clouds, full_joints, rotmats, body_label = [], [], [], []

    if 'full_joints' not in dataset or 'rotmats' not in dataset:
        flag = True
    else:
        flag = False

    limb_or_head = option[0]
    mask_or_noice = option[1]
    follow_or_static = option[2]

    #torch.set_printoptions(precision=20)
    for l in tqdm(range(0, B, bs)):
        r = l + bs
        if r > B: r = B
        span = r - l

        with torch.no_grad():

            p = torch.from_numpy(dataset['point_clouds'][l:r])

            follow_or_static = [True, False][random.randint(0, 1)]

            frame_l, frame_r = (0, 16) if follow_or_static else (7, 8)

            limb_or_head = ['limb', 'feet', 'head'][random.randint(0, 2)]

            if limb_or_head == 'limb':
                cp = (p[:, frame_l:frame_r, :, :].mean(dim=-2, keepdim=True) +
                      p[:, frame_l:frame_r, :, :].min(dim=-2, keepdim=True)[0]) / 2
            elif limb_or_head == 'feet':
                cp = p[:, frame_l:frame_r, :, :].min(dim=-2, keepdim=True)[0]
            else:
                cp = (p[:, frame_l:frame_r, :, :].mean(dim=-2, keepdim=True) +
                      p[:, frame_l:frame_r, :, :].max(dim=-2, keepdim=True)[0]) / 2
            angle = torch.sqrt((cp ** 2).sum(dim=-1)) / torch.sqrt((cp ** 2).sum(dim=-1) + 0.4 ** 2)
            cosine = torch.cosine_similarity(p, cp, dim=-1)
            mask = cosine > angle
            maskp = p.clone()
            if mask_or_noice:
                replacep = p[:, :, 0:1, :].expand(-1, -1, 512, -1)[mask].clone()
            else:
                replacep = maskp[mask] + (np.random.rand(*(maskp[mask]).shape) - 0.5) / 6

            maskp[mask] = replacep

            masked_point_clouds.append(maskp.cpu().numpy())
            body_label.append(np.logical_not(mask.cpu().numpy()))

            if flag:
                gt_rotmats = axis_angle_to_rotation_matrix(torch.from_numpy(dataset['pose'][l:r]).float().reshape(-1, 3)).reshape(-1, 24, 3, 3)
                gt_vertices = smpl(gt_rotmats.reshape(-1, 24, 3, 3).cuda(), torch.zeros((span * T, 10)).cuda())
                gt_full_joints = smpl.get_full_joints(gt_vertices).cpu().numpy().reshape(span, 16, 24, 3)

                gt_rotmats = gt_rotmats.cpu().float().numpy().reshape(-1, 16, 24, 3, 3)

                full_joints.append(gt_full_joints)
                rotmats.append(gt_rotmats)


    masked_point_clouds = np.concatenate(masked_point_clouds, axis=0)
    body_label = np.concatenate(body_label, axis=0)
    if flag:
        full_joints = np.concatenate(full_joints, axis=0)
        rotmats = np.concatenate(rotmats, axis=0)

    #prefix = '_'.join([limb_or_head, 'mask' if mask_or_noice else 'noice', 'follow' if follow_or_static else 'static'])
    #output_dataset_name = os.path.join(os.path.dirname(output_dataset_name), prefix + '_' +  os.path.basename(output_dataset_name))
    print(f'create dataset:{output_dataset_name}')
    with h5py.File(output_dataset_name, 'w') as f:
        f.create_dataset('masked_point_clouds', data=masked_point_clouds.reshape((-1, 512, 3)))
        f.create_dataset('body_label', data=body_label.reshape((-1, 512)))
        f.create_dataset('point_clouds', data=dataset['point_clouds'][:].reshape((-1, 512, 3)))
        f.create_dataset('points_num', data=dataset['points_num'][:].reshape((-1)))
        f.create_dataset('pose', data=dataset['pose'][:].reshape((-1, 72)))
        f.create_dataset('shape', data=dataset['shape'][:].reshape((-1, 10)))
        f.create_dataset('trans', data=dataset['trans'][:].reshape((-1, 3)))
        f.create_dataset('rotmats', data=dataset['rotmats'][:].reshape((-1, 24, 3, 3)))
        f.create_dataset('full_joints', data=dataset['full_joints'][:].reshape((-1, 24, 3)))

#gen_mask_dataset('/cwang/home/mqh/lidarcap_fusion/mask_lidarcap_train.hdf5', '/SAMSUMG8T/mqh/lidarcapv2/dataset/mask_limb_lidarcap_train.hdf5')
gen_mask_dataset('/cwang/home/mqh/lidarcap_fusion/mask_lidarcap_train.hdf5', '/SAMSUMG8T/mqh/lidarcapv2/dataset/mask_lidarcap_train2.hdf5', [None, False, None])

#gen_mask_dataset(args.in_path, args.out_path, ['feet', False, False])
"""
for i in [True, False]:
    for j in [True, False]:
        for k in [True, False]:
            gen_mask_dataset(args.in_path, args.out_path, [i, j, k])"""