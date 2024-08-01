# -*- coding: utf-8 -*-
# @Author  : jingyi
import os, sys
import argparse
import numpy as np
import torch

sys.path.append('/cwang/home/mqh/lidarcap_fusion/zjy/lidarcap_v2')
from modules import SMPL
from gen_dataset import *
from tools import multiprocess
from utils.save_visualization_file import save_ply
ROOT_PATH = '/SAMSUMG8T/ljl/zjy/sequences_downsample'
extras_path = '/SAMSUMG8T/mqh/lidarcapv2/behave'
MAX_PROCESS_COUNT = 32


def foo_(ids, npoints):
    ids = str(ids)

    smpl = SMPL()

    pose_filenames = os.path.join(ROOT_PATH, 'labels', '3d', 'pose', ids)
    json_filenames = list(filter(lambda x: x.endswith('pkl'), os.listdir(pose_filenames)))
    json_filenames = [os.path.join(pose_filenames, json_filename) for json_filename in json_filenames]

    cur_betas, cur_poses, cur_trans = multiprocess.multi_func(
        parse_pkl, MAX_PROCESS_COUNT, len(json_filenames), 'Load json files',
        True, json_filenames)

    segment_path = os.path.join(ROOT_PATH, 'labels', '3d', 'segment', id)
    segment_filenames = [os.path.join(segment_path, segment_filename) for segment_filename in os.listdir(segment_path)]
    cur_point_clouds = multiprocess.multi_func(
        read_point_cloud, MAX_PROCESS_COUNT, len(segment_filenames),
        'Load segment files', True, segment_filenames)

    cur_points_nums = [min(npoints, points.shape[0])
                       for points in cur_point_clouds]
    cur_point_clouds = [fix_points_num(
        points, npoints) for points in cur_point_clouds]

    poses = []
    betas = []
    trans = []
    # vertices = []
    points_nums = []
    point_clouds = []
    full_joints = []
    rotmats = []

    # assert(seqlen != 0)

    n = len(cur_betas)

    for i in range(n):
        # [lb, ub)
        # lb = i * seqlen
        # ub = lb + seqlen
        # img_filenames.append(cur_img_filenames[lb:ub])
        np_betas = np.stack(cur_betas[i])
        betas.append(np_betas)
        np_poses = np.stack(cur_poses[i])
        poses.append(np_poses)
        trans.append(np.stack(cur_trans[i]))
        # vertices.append(np.stack(cur_vertices[i]))
        point_clouds.append(np.stack(cur_point_clouds[i]))
        points_nums.append(cur_points_nums[i])
        #depths.append(cur_depths[lb:ub])

        #vertice = smpl(torch.from_numpy(np_poses), torch.zeros((seqlen, 10)))
        #x = save_ply(vertice[0]+trans[0],'test.ply')
        full_joints.append(smpl.get_full_joints(smpl(torch.from_numpy(np_poses[np.newaxis,:]), torch.from_numpy(np_betas[np.newaxis,:]))).cpu().numpy())

        rotmats.append(axis_angle_to_rotation_matrix(torch.from_numpy(np_poses.reshape(-1, 3))).reshape(24, 3, 3))

    # with open(os.path.join(os.path.dirname(ROOT_PATH), 'raw', 'process_info.json')) as f:
    #     process_json = json.load(f)
    # lidar_to_mocap_RT = np.array(process_json[str(ids)]['lidar_to_mocap_RT']).reshape(1, 4, 4).repeat(len(poses), axis=0)
    lidar_to_mocap_RT = np.zeros((0, 4, 4))

    # return poses, betas, trans, vertices, point_clouds, points_nums
    return np.stack(poses), np.stack(betas), np.stack(trans), np.stack(point_clouds), \
           np.stack(points_nums), np.stack(full_joints), lidar_to_mocap_RT,  np.stack(rotmats)


def dump_(ids, npoints):
    #ids = [1, 2, 3, 410, 4, 50301, 50302, 50304, 50305, 50306, 50307, 50308]

    whole_poses = np.zeros((0, 72))
    whole_betas = np.zeros((0, 10))
    whole_trans = np.zeros((0, 3))
    # whole_vertices = np.zeros((0, 6890, 3))
    whole_point_clouds = np.zeros((0, npoints, 3))
    whole_points_nums = np.zeros((0,))
    whole_full_joints = np.zeros((0, 24, 3))
    whole_lidar_to_mocap_RT = np.zeros((0, 4, 4))
    whole_rotmats = np.zeros((0, 24, 3, 3))
    #whole_depths = []

    print('start process', ids)

    poses, betas, trans, point_clouds, points_nums, full_joints, lidar_to_mocap_RT, rotmats = foo_(ids, npoints)

    whole_poses = np.concatenate((whole_poses, np.stack(poses)))
    whole_betas = np.concatenate((whole_betas, np.stack(betas)))
    whole_trans = np.concatenate((whole_trans, np.stack(trans)))
    # whole_vertices = np.concatenate(
    #     (whole_vertices, np.stack(vertices)))
    whole_point_clouds = np.concatenate(
        (whole_point_clouds, np.stack(point_clouds)))
    whole_points_nums = np.concatenate(
        (whole_points_nums, np.stack(points_nums)))
    #whole_depths += depths
    whole_full_joints = np.concatenate((whole_full_joints, full_joints.squeeze()))
    whole_lidar_to_mocap_RT = np.concatenate((whole_lidar_to_mocap_RT, lidar_to_mocap_RT))
    whole_rotmats = np.concatenate((whole_rotmats, rotmats))
    data = dict(whole_poses=whole_poses,
                whole_betas=whole_betas,
                whole_trans=whole_trans,
                whole_point_clouds=whole_point_clouds,
                whole_points_nums=whole_points_nums,
                whole_full_joints=whole_full_joints,
                whole_lidar_to_mocap_RT=whole_lidar_to_mocap_RT,
                whole_rotmats=whole_rotmats)
    return data

def save_hdf5(name,data):
    whole_filename = name + '.hdf5'
    with h5py.File(os.path.join(extras_path, whole_filename), 'w') as f:
        f.create_dataset('pose', data=data['whole_poses'])
        f.create_dataset('shape', data=data['whole_betas'])
        f.create_dataset('trans', data=data['whole_trans'])
        # f.create_dataset('human_vertex', data=whole_vertices)
        f.create_dataset('point_clouds', data=data['whole_point_clouds'])
        f.create_dataset('points_num', data=data['whole_points_nums'])
        #f.create_dataset('depth', data=whole_depths)
        f.create_dataset('full_joints', data=data['whole_full_joints'])
        f.create_dataset('lidar_to_mocap_RT', data=data['whole_lidar_to_mocap_RT'])
        f.create_dataset('rotmats', data=data['whole_rotmats'])

    print('Success create dataset:', os.path.join(extras_path, whole_filename))

if __name__ == '__main__':
    low_train = ['Date03_Sub03_trashbin', 'Date03_Sub04_basketball',
                 'Date03_Sub04_boxsmall', 'Date03_Sub04_boxtiny',
                 'Date03_Sub04_keyboard_move', 'Date03_Sub04_keyboard_typing',
                 'Date03_Sub04_toolbox', 'Date03_Sub04_trashbin',
                 'Date03_Sub05_basketball',
                 'Date03_Sub05_boxsmall', 'Date03_Sub05_boxtiny',
                 'Date03_Sub05_keyboard',
                 'Date03_Sub05_toolbox', 'Date03_Sub05_trashbin',
                 'Date04_Sub05_basketball',
                 'Date04_Sub05_boxsmall', 'Date04_Sub05_boxtiny',
                 'Date04_Sub05_keyboard',
                 'Date04_Sub05_toolbox', 'Date04_Sub05_trashbin',
                 'Date05_Sub06_basketball',
                 'Date05_Sub06_boxsmall', 'Date05_Sub06_boxtiny',
                 'Date05_Sub06_keyboard_hand', 'Date05_Sub06_keyboard_move',
                 'Date05_Sub06_toolbox', 'Date05_Sub06_trashbin',
                 'Date06_Sub07_basketball',
                 'Date06_Sub07_boxsmall', 'Date06_Sub07_boxtiny',
                 'Date06_Sub07_keyboard_move', 'Date06_Sub07_keyboard_typing',
                 'Date06_Sub07_toolbox', 'Date06_Sub07_trashbin',
                 'Date07_Sub04_basketball',
                 'Date07_Sub04_boxsmall', 'Date07_Sub04_boxtiny',
                 'Date07_Sub04_keyboard_move', 'Date07_Sub04_keyboard_typing',
                 'Date07_Sub04_toolbox_lift', 'Date07_Sub04_trashbin',
                 'Date07_Sub08_basketball', 'Date07_Sub08_boxsmall',
                 'Date07_Sub08_boxtiny',
                 'Date07_Sub08_keyboard_move', 'Date07_Sub08_keyboard_typing',
                 'Date07_Sub08_toolbox', 'Date07_Sub08_trashbin']
    mid_train = ['Date03_Sub03_stool_sit', 'Date03_Sub03_suitcase_lift',
                 'Date03_Sub03_suitcase_move', 'Date03_Sub04_backpack_back',
                 'Date03_Sub04_backpack_hand', 'Date03_Sub04_backpack_hug',
                 'Date03_Sub04_boxmedium', 'Date03_Sub04_plasticcontainer_lift',
                 'Date03_Sub04_stool_move', 'Date03_Sub04_stool_sit',
                 'Date03_Sub04_suitcase_ground', 'Date03_Sub04_suitcase_lift',
                 'Date03_Sub05_backpack', 'Date03_Sub05_boxmedium',
                 'Date03_Sub05_plasticcontainer', 'Date03_Sub05_stool',
                 'Date03_Sub05_suitcase', 'Date04_Sub05_backpack',
                 'Date04_Sub05_boxmedium',
                 'Date04_Sub05_plasticcontainer', 'Date04_Sub05_stool',
                 'Date04_Sub05_suitcase', 'Date04_Sub05_suitcase_open',
                 'Date05_Sub05_backpack', 'Date05_Sub06_backpack_back',
                 'Date05_Sub06_backpack_hand', 'Date05_Sub06_backpack_twohand',
                 'Date05_Sub06_boxmedium', 'Date05_Sub06_plasticcontainer',
                 'Date05_Sub06_stool_lift', 'Date05_Sub06_stool_sit',
                 'Date05_Sub06_suitcase_hand', 'Date05_Sub06_suitcase_lift',
                 'Date06_Sub07_backpack_back', 'Date06_Sub07_backpack_hand',
                 'Date06_Sub07_backpack_twohand', 'Date06_Sub07_boxmedium',
                 'Date06_Sub07_plasticcontainer', 'Date06_Sub07_stool_lift',
                 'Date06_Sub07_stool_sit', 'Date06_Sub07_suitcase_lift',
                 'Date06_Sub07_suitcase_move', 'Date07_Sub04_backpack_back',
                 'Date07_Sub04_backpack_hand', 'Date07_Sub04_backpack_twohand',
                 'Date07_Sub04_boxmedium', 'Date07_Sub04_plasticcontainer',
                 'Date07_Sub04_stool_lift', 'Date07_Sub04_stool_sit',
                 'Date07_Sub04_suitcase_lift', 'Date07_Sub04_suitcase_open',
                 'Date07_Sub05_suitcase_lift', 'Date07_Sub05_suitcase_open',
                 'Date07_Sub08_backpack_back', 'Date07_Sub08_backpack_hand',
                 'Date07_Sub08_backpack_hug', 'Date07_Sub08_boxmedium',
                 'Date07_Sub08_plasticcontainer', 'Date07_Sub08_stool',
                 'Date07_Sub08_suitcase']
    hard_train = ['Date03_Sub03_tablesmall_move', 'Date03_Sub03_tablesquare_lift',
                  'Date03_Sub03_tablesquare_move', 'Date03_Sub03_tablesquare_sit',
                  'Date03_Sub03_yogaball_play', 'Date03_Sub03_yogaball_sit',
                  'Date03_Sub03_yogamat', 'Date03_Sub04_boxlarge',
                  'Date03_Sub04_boxlong',
                  'Date03_Sub04_chairblack_hand', 'Date03_Sub04_chairblack_liftreal',
                  'Date03_Sub04_chairblack_sit', 'Date03_Sub04_chairwood_hand',
                  'Date03_Sub04_chairwood_lift', 'Date03_Sub04_chairwood_sit',
                  'Date03_Sub04_monitor_hand', 'Date03_Sub04_monitor_move',
                  'Date03_Sub04_tablesmall_hand', 'Date03_Sub04_tablesmall_lean',
                  'Date03_Sub04_tablesmall_lift', 'Date03_Sub04_tablesquare_hand',
                  'Date03_Sub04_tablesquare_lift', 'Date03_Sub04_tablesquare_sit',
                  'Date03_Sub04_yogaball_play', 'Date03_Sub04_yogaball_sit',
                  'Date03_Sub04_yogamat', 'Date03_Sub05_boxlarge',
                  'Date03_Sub05_boxlong',
                  'Date03_Sub05_chairblack', 'Date03_Sub05_chairwood',
                  'Date03_Sub05_monitor',
                  'Date03_Sub05_tablesmall', 'Date03_Sub05_tablesquare',
                  'Date03_Sub05_yogaball', 'Date03_Sub05_yogamat',
                  'Date04_Sub05_boxlarge',
                  'Date04_Sub05_boxlong', 'Date04_Sub05_chairblack',
                  'Date04_Sub05_chairwood',
                  'Date04_Sub05_monitor', 'Date04_Sub05_monitor_sit',
                  'Date04_Sub05_tablesmall', 'Date04_Sub05_tablesquare',
                  'Date04_Sub05_yogaball', 'Date04_Sub05_yogamat',
                  'Date05_Sub05_chairblack',
                  'Date05_Sub05_chairwood', 'Date05_Sub05_yogaball',
                  'Date05_Sub06_boxlarge',
                  'Date05_Sub06_boxlong', 'Date05_Sub06_chairblack_hand',
                  'Date05_Sub06_chairblack_lift', 'Date05_Sub06_chairblack_sit',
                  'Date05_Sub06_chairwood_hand', 'Date05_Sub06_chairwood_lift',
                  'Date05_Sub06_chairwood_sit', 'Date05_Sub06_monitor_hand',
                  'Date05_Sub06_monitor_move', 'Date05_Sub06_tablesmall_hand',
                  'Date05_Sub06_tablesmall_lean', 'Date05_Sub06_tablesmall_lift',
                  'Date05_Sub06_tablesquare_lift', 'Date05_Sub06_tablesquare_move',
                  'Date05_Sub06_tablesquare_sit', 'Date05_Sub06_yogaball_play',
                  'Date05_Sub06_yogaball_sit', 'Date05_Sub06_yogamat',
                  'Date06_Sub07_boxlarge', 'Date06_Sub07_boxlong',
                  'Date06_Sub07_chairblack_hand', 'Date06_Sub07_chairblack_lift',
                  'Date06_Sub07_chairblack_sit', 'Date06_Sub07_chairwood_hand',
                  'Date06_Sub07_chairwood_lift', 'Date06_Sub07_chairwood_sit',
                  'Date06_Sub07_monitor_move', 'Date06_Sub07_tablesmall_lean',
                  'Date06_Sub07_tablesmall_lift', 'Date06_Sub07_tablesmall_move',
                  'Date06_Sub07_tablesquare_lift', 'Date06_Sub07_tablesquare_move',
                  'Date06_Sub07_tablesquare_sit', 'Date06_Sub07_yogaball_play',
                  'Date06_Sub07_yogaball_sit', 'Date06_Sub07_yogamat',
                  'Date07_Sub04_boxlarge', 'Date07_Sub04_boxlong',
                  'Date07_Sub04_chairblack_hand', 'Date07_Sub04_chairblack_lift',
                  'Date07_Sub04_chairblack_sit', 'Date07_Sub04_chairwood_hand',
                  'Date07_Sub04_chairwood_lift', 'Date07_Sub04_chairwood_sit',
                  'Date07_Sub04_monitor_hand', 'Date07_Sub04_monitor_move',
                  'Date07_Sub04_tablesmall_lean', 'Date07_Sub04_tablesmall_lift',
                  'Date07_Sub04_tablesmall_move', 'Date07_Sub04_tablesquare_lift',
                  'Date07_Sub04_tablesquare_move', 'Date07_Sub04_tablesquare_sit',
                  'Date07_Sub04_yogaball_play', 'Date07_Sub04_yogaball_sit',
                  'Date07_Sub04_yogamat', 'Date07_Sub05_tablesmall',
                  'Date07_Sub05_tablesquare', 'Date07_Sub08_boxlarge',
                  'Date07_Sub08_boxlong',
                  'Date07_Sub08_chairblack_hand', 'Date07_Sub08_chairblack_lift',
                  'Date07_Sub08_chairblack_sit', 'Date07_Sub08_chairwood_hand',
                  'Date07_Sub08_chairwood_lift', 'Date07_Sub08_chairwood_sit',
                  'Date07_Sub08_monitor_hand', 'Date07_Sub08_monitor_move',
                  'Date07_Sub08_tablesmall', 'Date07_Sub08_tablesquare',
                  'Date07_Sub08_yogaball', 'Date07_Sub08_yogamat']
    train = low_train + mid_train + hard_train

    low_test = ['Date01_Sub01_basketball', 'Date01_Sub01_boxsmall_hand',
                'Date01_Sub01_boxtiny_hand', 'Date01_Sub01_keyboard_move',
                'Date01_Sub01_keyboard_typing', 'Date01_Sub01_toolbox',
                'Date01_Sub01_trashbin', 'Date02_Sub02_basketball',
                'Date02_Sub02_boxsmall_hand', 'Date02_Sub02_boxtiny_hand',
                'Date02_Sub02_keyboard_move', 'Date02_Sub02_keyboard_typing',
                'Date02_Sub02_toolbox', 'Date02_Sub02_trashbin',
                'Date03_Sub03_basketball',
                'Date03_Sub03_boxsmall', 'Date03_Sub03_boxtiny',
                'Date03_Sub03_keyboard_move', 'Date03_Sub03_keyboard_typing',
                'Date03_Sub03_toolbox']
    mid_test = ['Date01_Sub01_backpack_back', 'Date01_Sub01_backpack_hand',
                'Date01_Sub01_backpack_hug', 'Date01_Sub01_boxmedium_hand',
                'Date01_Sub01_plasticcontainer', 'Date01_Sub01_stool_move',
                'Date01_Sub01_stool_sit', 'Date01_Sub01_suitcase',
                'Date01_Sub01_suitcase_lift', 'Date02_Sub02_backpack_back',
                'Date02_Sub02_backpack_hand', 'Date02_Sub02_backpack_twohand',
                'Date02_Sub02_boxmedium_hand', 'Date02_Sub02_plasticcontainer',
                'Date02_Sub02_stool_move', 'Date02_Sub02_stool_sit',
                'Date02_Sub02_suitcase_ground', 'Date02_Sub02_suitcase_lift',
                'Date03_Sub03_backpack_back', 'Date03_Sub03_backpack_hand',
                'Date03_Sub03_backpack_hug', 'Date03_Sub03_boxmedium',
                'Date03_Sub03_plasticcontainer', 'Date03_Sub03_stool_lift', ]
    hard_test = ['Date01_Sub01_boxlarge_hand', 'Date01_Sub01_boxlong_hand',
                 'Date01_Sub01_chairblack_hand', 'Date01_Sub01_chairblack_lift',
                 'Date01_Sub01_chairblack_sit', 'Date01_Sub01_chairwood_hand',
                 'Date01_Sub01_chairwood_lift', 'Date01_Sub01_chairwood_sit',
                 'Date01_Sub01_monitor_hand', 'Date01_Sub01_monitor_move',
                 'Date01_Sub01_tablesmall_lean', 'Date01_Sub01_tablesmall_lift',
                 'Date01_Sub01_tablesmall_move', 'Date01_Sub01_tablesquare_hand',
                 'Date01_Sub01_tablesquare_lift', 'Date01_Sub01_tablesquare_sit',
                 'Date01_Sub01_yogaball', 'Date01_Sub01_yogaball_play',
                 'Date01_Sub01_yogamat_hand', 'Date02_Sub02_boxlarge_hand',
                 'Date02_Sub02_boxlong_hand', 'Date02_Sub02_chairblack_hand',
                 'Date02_Sub02_chairblack_lift', 'Date02_Sub02_chairblack_sit',
                 'Date02_Sub02_chairwood_hand', 'Date02_Sub02_chairwood_sit',
                 'Date02_Sub02_monitor_hand', 'Date02_Sub02_monitor_move',
                 'Date02_Sub02_tablesmall_lean', 'Date02_Sub02_tablesmall_lift',
                 'Date02_Sub02_tablesmall_move', 'Date02_Sub02_tablesquare_lift',
                 'Date02_Sub02_tablesquare_move', 'Date02_Sub02_tablesquare_sit',
                 'Date02_Sub02_yogaball_play', 'Date02_Sub02_yogaball_sit',
                 'Date02_Sub02_yogamat', 'Date03_Sub03_boxlarge',
                 'Date03_Sub03_boxlong',
                 'Date03_Sub03_chairblack_hand', 'Date03_Sub03_chairblack_lift',
                 'Date03_Sub03_chairblack_sit', 'Date03_Sub03_chairblack_sitstand',
                 'Date03_Sub03_chairwood_hand', 'Date03_Sub03_chairwood_lift',
                 'Date03_Sub03_chairwood_sit', 'Date03_Sub03_monitor_move',
                 'Date03_Sub03_tablesmall_lean', 'Date03_Sub03_tablesmall_lift']

    # os.makedirs(extras_path, exist_ok=True)

    parser = argparse.ArgumentParser()

    #parser.add_argument('--seqlen', type=int, default=16)
    parser.add_argument('--npoints', type=int, default=512)
    parser.add_argument('--gpu', type=int, required=True)
    parser.set_defaults(func=dump_)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    _content = {}
    for id in train:
        content = dump_(id, args.npoints)
        if len(_content.keys()) == 0:
            _content = content
        else:
            for k, v in content.items():
                _content[k]= np.concatenate((_content[k], content[k]))
    save_hdf5('BEHAVE_train', _content)

    # ------------------------------------------------

    _content={}
    for id in low_test:
        content = dump_(id, args.npoints)
        if len(_content.keys()) == 0:
            _content = content
        else:
            for k, v in content.items():
                _content[k] = np.concatenate((_content[k], content[k]))
    save_hdf5('BEHAVE_low_test', _content)

    # ------------------------------------------------

    _content={}
    for id in mid_test:
        content = dump_(id, args.npoints)
        if len(_content.keys()) == 0:
            _content = content
        else:
            for k, v in content.items():
                _content[k] = np.concatenate((_content[k], content[k]))
    save_hdf5('BEHAVE_mid_test', _content)

    # ------------------------------------------------

    _content={}
    for id in hard_test:
        content = dump_(id, args.npoints)
        if len(_content.keys()) == 0:
            _content = content
        else:
            for k, v in content.items():
                _content[k] = np.concatenate((_content[k], content[k]))
    save_hdf5('BEHAVE_hard_test', _content)
