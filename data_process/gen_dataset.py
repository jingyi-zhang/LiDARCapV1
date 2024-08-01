from ast import parse
from plyfile import PlyData, PlyElement
from typing import List
import argparse
import numpy as np
import json
import os
import re
import sys
import h5py
import torch
import pickle as pkl

from tqdm import tqdm
import sys
sys.path.append('/cwang/home/mqh/lidarcapv2')
from modules.geometry import axis_angle_to_rotation_matrix

from tools import multiprocess
from modules.smpl import SMPL
from scipy.spatial.transform import Rotation as R


ROOT_PATH = '/SAMSUMG8T/mqh/lidarcapv2/lidarcap'
MAX_PROCESS_COUNT = 32

# img_filenames = []

import open3d as o3d

def read_point_cloud(filename):
    return np.asarray(o3d.io.read_point_cloud(filename).points)

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    ply_data = PlyData.read(filename)['vertex'].data
    points = np.array([[x, y, z] for x, y, z in ply_data])
    return points


def save_ply(filename, points):
    points = [(points[i, 0], points[i, 1], points[i, 2])
              for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=False).write(filename)


def get_index(filename):
    basename = os.path.basename(filename)
    return int(os.path.splitext(basename)[0])


def get_sorted_filenames_by_index(dirname, isabs=True):
    if not os.path.exists(dirname):
        return []
    filenames = os.listdir(dirname)
    filenames = sorted(os.listdir(dirname), key=lambda x: get_index(x))
    if isabs:
        filenames = [os.path.join(dirname, filename) for filename in filenames]
    return filenames

def get_paths_by_suffix(dirname, suffix):
    filenames = list(filter(lambda x: x.endswith(suffix), os.listdir(dirname)))
    assert len(filenames) > 0
    return [os.path.join(dirname, filename) for filename in filenames]

def parse_json(json_filename):
    with open(json_filename) as f:
        content = json.load(f)
        beta = np.array(content['beta'], dtype=np.float32)
        pose = np.array(content['pose'], dtype=np.float32)
        trans = np.array(content['trans'], dtype=np.float32)
    return beta, pose, trans

def parse_pkl(pkl_filename):
    mocap2lidar_matrix = np.array([
        [-1,0,0],
        [0,0,1],
        [0,1,0]])

    behave2lidarcap = np.array([
        [ 0.34114292, -0.81632519,  0.46608443],
        [-0.93870972, -0.26976026,  0.21460074],
        [-0.04945297, -0.5107275, -0.85831922]])
    with open(pkl_filename, 'rb') as f:
        content = pkl.load(f)
        beta = np.array(content['betas'], dtype=np.float32) # (10, )
        pose = np.array(content['pose'][:72], dtype=np.float32) # (72, )
        trans = np.array(content['trans'], dtype=np.float32)# (3, )
    pose[:3]=(R.from_matrix(mocap2lidar_matrix) * R.from_rotvec(pose[:3])).as_rotvec()
    pose[:3]=(R.from_matrix(behave2lidarcap) * R.from_rotvec(pose[:3])).as_rotvec()
    return beta, pose, trans


def fix_points_num(points: np.array, num_points: int):
    points = points[~np.isnan(points).any(axis=-1)]

    origin_num_points = points.shape[0]
    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    if origin_num_points >= num_points:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res

def compute_dist(a, b):
    pcda = o3d.geometry.PointCloud()
    pcda.points = o3d.utility.Vector3dVector(a)
    pcdb = o3d.geometry.PointCloud()
    pcdb.points = o3d.utility.Vector3dVector(b)
    return np.asarray(pcda.compute_point_cloud_distance(pcdb))

mqh_shape = np.array([float(e) for e in "0.01938907 -0.27631596 0.07859833 0.1386834 -0.20010415 -0.02591152 0.23835489 -0.00451579 -0.00132555 0.01529741".split(' ')])
dyd_shape = np.array([float(e) for e in "0.13718624 -0.32368565 0.06066366 0.22490674 -0.3380233 -0.1569234 0.32280767 -0.00115923 -0.04938826 0.04286334".split(' ')])
zjy_shape = np.array([1.14969783e-01, -1.80414230e-01, 5.12518398e-02, 2.13451669e-01, -3.37647825e-01, -4.28195596e-02,2.22227722e-01, 6.78026900e-02, 2.21031010e-02, 7.61183398e-03])


def foo(id, npoints):
    id = str(id)

    smpl = SMPL()

    # cur_img_filenames = get_sorted_filenames_by_index(
    #     os.path.join(ROOT_PATH, 'images', id))

    pose_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'pose', id))
    json_filenames = list(filter(lambda x: x.endswith('json'), pose_filenames))
    #ply_filenames = list(filter(lambda x: x.endswith('ply'), pose_filenames))

    cur_betas, cur_poses, cur_trans = multiprocess.multi_func(
        parse_json, MAX_PROCESS_COUNT, len(json_filenames), 'Load json files',
        True, json_filenames)
    # cur_vertices = multiprocess.multi_func(
    #     read_ply, MAX_PROCESS_COUNT, len(ply_filenames), 'Load vertices files',
    #     True, ply_filenames)

    depth_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'depth', id))
    cur_depths = depth_filenames

    segment_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'segment', id))
    cur_point_clouds = multiprocess.multi_func(
        read_point_cloud, MAX_PROCESS_COUNT, len(segment_filenames),
        'Load segment files', True, segment_filenames)

    cur_points_nums = [min(npoints, points.shape[0]) for points in cur_point_clouds]
    cur_point_clouds = [fix_points_num(points, npoints) for points in cur_point_clouds]

    poses = []
    betas = []
    trans = []
    # vertices = []
    points_nums = []
    point_clouds = []
    depths = []
    full_joints = []
    rotmats = []
    body_mask = []

    # assert(seqlen != 0)

    n = len(cur_betas)
    # 直接补齐
    # while n % seqlen != 0:
    #     # cur_img_filenames.append(cur_img_filenames[-1])
    #     cur_betas.append(cur_betas[-1])
    #     cur_poses.append(cur_poses[-1])
    #     cur_trans.append(cur_trans[-1])
    #     # cur_vertices.append(cur_vertices[-1])
    #     cur_point_clouds.append(cur_point_clouds[-1])
    #     cur_points_nums.append(cur_points_nums[-1])
    #     #cur_depths.append(cur_depths[-1])
    #     n += 1
    # times = n // seqlen
    torch.set_num_threads(1)
    for i in tqdm(range(len(cur_poses))):
        # [lb, ub)
        # lb = i * seqlen
        # ub = lb + seqlen
        # img_filenames.append(cur_img_filenames[lb:ub])
        np_betas = np.stack(cur_betas[i])
        betas.append(np_betas)
        np_poses = np.stack(cur_poses[i])
        poses.append(np_poses)
        trans.append(np.stack(cur_trans[i]))
        # vertices.append(np.stack(cur_vertices[lb:ub]))
        point_clouds.append(np.stack(cur_point_clouds[i]))
        points_nums.append(cur_points_nums[i])
        #depths.append(cur_depths[lb:ub])

        v = smpl(torch.from_numpy(np_poses[np.newaxis, :]), torch.from_numpy(np_betas[np.newaxis, :]))
        full_joints.append(smpl.get_full_joints(v).cpu().numpy())

        rotmats.append(axis_angle_to_rotation_matrix(torch.from_numpy(np_poses.reshape(-1, 3))).reshape(24, 3, 3))

        pc = cur_point_clouds[i] - cur_trans[i]
        d = compute_dist(pc, v[0].cpu().numpy())
        body_mask.append(d < 0.1)
        #pc2 = pc.copy()
        #pc2[~body_mask[-1]] = 0
        #ovis.magic((rotmats[-1], np.arange(i, i + 1)), (pc + np.array([[0, -0.6, 0]]), np.arange(i, i + 1)), (pc2 + np.array([[0, 0.6, 0]]), np.arange(i, i + 1)))

    with open(os.path.join(os.path.dirname(ROOT_PATH), 'raw', 'process_info.json')) as f:
        process_json = json.load(f)
    lidar_to_mocap_RT = np.array(process_json[str(id)]['lidar_to_mocap_RT']).reshape(1, 4, 4).repeat(len(poses), axis=0)

    poses, betas, trans, point_clouds, points_nums, depths, full_joints, lidar_to_mocap_RT, rotmats, body_mask = \
        np.stack(poses), np.stack(betas), np.stack(trans), np.stack(point_clouds), np.stack(points_nums), depths, np.stack(full_joints), lidar_to_mocap_RT,  np.stack(rotmats), np.stack(body_mask)


    return poses, betas, trans, point_clouds, points_nums, depths, full_joints, lidar_to_mocap_RT, rotmats, body_mask


def test(args):
    pass


def get_sorted_ids(s):
    if re.match('^([1-9]\d*)-([1-9]\d*)$', s):
        start_index, end_index = s.split('-')
        indexes = list(range(int(start_index), int(end_index) + 1))
    elif re.match('^(([1-9]\d*),)*([1-9]\d*)$', s):
        indexes = [int(x) for x in s.split(',')]
    return sorted(indexes)


def dump(ids, npoints, name):
    #ids = [1, 2, 3, 410, 4, 50301, 50302, 50304, 50305, 50306, 50307, 50308]

    whole_poses = np.zeros((0, 72))
    whole_betas = np.zeros((0, 10))
    whole_trans = np.zeros((0, 3))
    # whole_vertices = np.zeros((0, 6890, 3))
    whole_point_clouds = np.zeros((0, npoints, 3))
    whole_points_nums = np.zeros((0))
    whole_full_joints = np.zeros((0, 24, 3))
    whole_lidar_to_mocap_RT = np.zeros((0, 4, 4))
    whole_rotmats = np.zeros((0, 24, 3, 3))
    whole_body_label = np.zeros((0, 512))
    #whole_depths = []

    for id in ids:
        print('start process', id)

        poses, betas, trans, point_clouds, points_nums, depths, full_joints, lidar_to_mocap_RT, rotmats, body_mask = foo(id, npoints)
        """
        import OVis
        human_points = point_clouds.copy()
        human_points[np.logical_not(body_mask)] = 0
        OVis.ovis.magic((rotmats, trans), (point_clouds + 0.5), (human_points - 0.5))"""

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
        whole_body_label = np.concatenate((whole_body_label, body_mask))

    whole_filename = name + '.hdf5'
    with h5py.File(os.path.join(extras_path, whole_filename), 'w') as f:
        f.create_dataset('pose', data=whole_poses)
        f.create_dataset('shape', data=whole_betas)
        f.create_dataset('trans', data=whole_trans)
        # f.create_dataset('human_vertex', data=whole_vertices)
        f.create_dataset('point_clouds', data=whole_point_clouds)
        f.create_dataset('points_num', data=whole_points_nums)
        #f.create_dataset('depth', data=whole_depths)
        f.create_dataset('full_joints', data=whole_full_joints)
        f.create_dataset('lidar_to_mocap_RT', data=whole_lidar_to_mocap_RT)
        f.create_dataset('rotmats', data=whole_rotmats)
        f.create_dataset('body_label', data=whole_body_label)

    print('Success create dataset:', os.path.join(extras_path, whole_filename))


if __name__ == '__main__':
    extras_path = '/SAMSUMG8T/mqh/lidarcapv2/dataset'
    os.makedirs(extras_path, exist_ok=True)
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    parser = argparse.ArgumentParser()

    # parser.add_argument('--seqlen', type=int, default=16)
    parser.add_argument('--npoints', type=int, default=512)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--gpu', type=int, required=True)
    parser.set_defaults(func=dump)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    for id in args.ids:
        dump([id, ], args.npoints, str(id))


