import numpy as np
import torch
import h5py
from torch.utils.data import Dataset

import os

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from modules.geometry import axis_angle_to_rotation_matrix
import pyransac3d as pyrsc
from yacs.config import CfgNode

def pc_normalize(pc):
    pc[..., 0:2] -= np.mean(pc[..., 0:2], axis=1, keepdims=True)
    pc[..., 2] -= np.mean(pc[..., 2])
    pc /= 1.2
    return pc

def pc_normalize_w_raw_z(pc):
    pc[..., 0:2] -= np.mean(pc[..., 0:2], axis=1, keepdims=True)
    #pc[..., 2] -= np.mean(pc[..., 2])
    #pc /= 1.2
    return pc


def augment(points, points_num):  # (T, N, 3), (T, )
    T, N = points.shape[:2]
    augmented_points = points.copy()

    # 放缩
    scale = np.random.uniform(0.9, 1.1)
    augmented_points *= scale

    # 随机丢弃，至少保留50个点
    dropout_ratio = np.clip(0, np.random.random() *
                            (1 - 50 / np.min(points_num)), 0.5)
    drop_idx = np.where(np.random.random((T, N)) <= dropout_ratio)
    augmented_points[drop_idx] = augmented_points[0][0]

    # 高斯噪声
    jittered_points = np.clip(
        0.01 * np.random.randn(*augmented_points.shape), -0.05, 0.05)
    augmented_points += jittered_points

    return augmented_points

def affine(X, matrix):
    if type(X) == np.ndarray:
        res = np.concatenate((X, np.ones((*X.shape[:-1], 1))), axis=-1).T
        res = np.dot(matrix, res).T
    else:
        res = torch.cat((X, torch.ones((*X.shape[:-1], 1)).to(X.device)), axis=-1)
        res = matrix.to(X.device).matmul(res.transpose(1, 2)).transpose(1, 2)
    return res[..., :-1]


class Lidarcapv2_Dataset(Dataset):
    default_cfg = {
        'dataset_path': 'your_data_path',
        'use_aug': False,
        'use_rot': False,
        'use_straight': False,
        'use_pc_w_raw_z': False,
        'ret_raw_pc': False,
        'seqlen': 16,
        'drop_first_n': 0,
        'add_noice_pc': False,
        'noice_pc_scale': 1.5,
        'set_body_label_all_one': False,
        'noice_pc_rate': 1.0,
        'replace_noice_pc': False,
        'replace_noice_pc_rate': 0.2,
        'random_permutation': False,
        'use_trans_to_normalize': False
    }

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        if cfg is not None:
            assert not hasattr(self, 'cfg'), 'cfg仅仅用于初始化！'
            self.cfg = cfg
            self.cfg.update({k: v for k, v in Lidarcapv2_Dataset.default_cfg.items() if k not in self.cfg})
        else:
            cfg = Lidarcapv2_Dataset.default_cfg.copy()
            cfg.update(kwargs)
            self.cfg = CfgNode(cfg)

        self.update_cfg()

    def update_cfg(self, **kwargs):
        assert all([k in self.cfg for k in kwargs.keys()])
        self.cfg.update(kwargs)

        self.dataset_path = self.cfg.dataset_path
        self.dataset_ids = self.cfg.dataset_ids


        self.length = 0
        self.lidar_to_mocap_RT_flag = True
        for id in self.dataset_ids:
            with h5py.File(os.path.join(self.dataset_path, f'{id}.hdf5'), 'r') as f:
                assert f['pose'][0].shape == (72, ), f"数据集：{os.path.join(self.dataset_path, f'{id}.hdf5')}中pose的shape:{f['pose'][0].shape}不正确！"
                self.length += (len(f['pose']) - self.cfg.drop_first_n) // self.cfg.seqlen
                if 'lidar_to_mocap_RT' not in f:
                    self.lidar_to_mocap_RT_flag = False

        if self.cfg.use_rot or self.cfg.use_straight:
            from modules import SMPL
            self.smpl = SMPL()
            #self.smpl = SMPL()
        else:
            self.lidar_to_mocap_RT_flag = False

    def __del__(self):
        if hasattr(self, 'datas'):
            for data in self.datas:
                data.close()
            print('success close dataset')

    def open_hdf5(self):
        self.datas = []
        self.datas_length = []
        #for id in tqdm(self.dataset_ids, desc="Load Datasets", ncols=60):

        for id in self.dataset_ids:
            f = h5py.File(os.path.join(self.dataset_path, f'{id}.hdf5'), 'r')
            self.datas_length.append(len(f['pose']))
            self.datas.append(f)

    def access_hdf5(self, index):
        seqlen = self.cfg.seqlen
        raw_index = int(index)
        for data, length in zip(self.datas, self.datas_length):
            dataset_max_index = (length - self.cfg.drop_first_n) // seqlen - 1
            if index > dataset_max_index:
                index -= dataset_max_index + 1
            else:
                l = self.cfg.drop_first_n + index * seqlen
                r = l + seqlen
                assert r <= len(data['pose']), 'access_hdf5：未知错误！'

                pose = data['pose'][l:r]
                betas = data['shape'][l:r]
                trans = data['trans'][l:r]
                if 'masked_point_clouds' in data:
                    human_points = data['masked_point_clouds'][l:r]
                else:
                    human_points = data['point_clouds'][l:r]
                points_num = data['points_num'][l:r]
                full_joints = data['full_joints'][l:r]
                rotmats = data['rotmats'][l:r]
                if self.lidar_to_mocap_RT_flag:
                    if 'lidar_to_mocap_RT' in data:
                        lidar_to_mocap_RT = data['lidar_to_mocap_RT'][l:r]
                    else:
                        assert self.cfg.use_rot is False, f'[ERROR]数据集{data}没有lidar_to_mocap_RT! use_rot选项只支持带有lidar_to_mocap_RT的数据集。'
                        lidar_to_mocap_RT = None
                else:
                    lidar_to_mocap_RT = None

                body_label = data['body_label'][l:r] if 'body_label' in data else None

                assert pose.shape == (seqlen, 72) and full_joints.shape == (seqlen, 24, 3) and rotmats.shape == (seqlen, 24, 3, 3) and human_points.shape == (seqlen, 512, 3), 'shape 不正确！'

                return pose, betas, trans, human_points, points_num, full_joints, rotmats, lidar_to_mocap_RT, body_label
        assert False, f'找不到index：{raw_index}对应的dataset'

    def access_hdf5_dataset(self, dataset_id, dataset_key):
        index = self.dataset_ids.index(dataset_id)
        if index < 0:
            print(f'找不到dataset_id:{dataset_id}！')
            return None
        return self.datas[index][dataset_key]

    def acquire_hdf5_by_index(self, index):
        for i, (data, length) in enumerate(zip(self.datas, self.datas_length)):
            if index > length - 1:
                index -= length
            else:
                return i, data

    def split_list_by_dataset(self, *l):
        left_i = 0
        seqlen = self.cfg.seqlen
        for i, length in enumerate(self.datas_length):
            #ret.append([e[left_i:left_i+length] for e in l])
            dataset_seq_count = (length - self.cfg.drop_first_n) // seqlen
            dataset_length = dataset_seq_count * seqlen
            yield [self.dataset_ids[i], ] + [e[left_i:left_i+dataset_length] for e in l]
            left_i += dataset_length

        assert all([len(e) == left_i for e in l]), 'assert false:split_list_by_dataset'

    def rot(self, data):
        pose = data['pose'].clone().reshape(1, 16, 72)
        rotmats = data['rotmats'].clone().reshape(1, 16, 24, 3, 3)

        B, T, _ = pose.shape
        lidar_to_mocap_RT = data['lidar_to_mocap_RT'].clone().reshape(1, 4, 4)
        lidar_to_mocap_RT[:, :3, 3] = 0
        mocap_to_lidar_RT = torch.from_numpy(np.linalg.inv(lidar_to_mocap_RT))

        angel = np.random.randint(1, 359)
        z_rot = np.eye(4)
        z_rot[:3, :3] = pyrsc.get_rotationMatrix_from_vectors([1, 0, 0], [np.cos(angel), np.sin(angel), 0])

        if self.cfg.use_rot:
            lidar_to_mocap_RT = torch.from_numpy(z_rot).reshape(1, 4, 4).float().matmul(lidar_to_mocap_RT)

        if not self.cfg.use_straight:
            lidar_to_mocap_RT = mocap_to_lidar_RT.matmul(lidar_to_mocap_RT)

        lidar_to_mocap_R = lidar_to_mocap_RT[:, :3, :3].reshape(B, 1, 3, 3).expand(-1, 16, -1, -1).reshape(B * T, 3, 3)
        pose = pose.reshape(B * T, 72)
        pose[:, :3] = torch.from_numpy((R.from_matrix(lidar_to_mocap_R) * R.from_rotvec(pose[:, :3])).as_rotvec())
        assert not (torch.any(torch.isnan(pose)) or torch.any(torch.isinf(pose)))
        rotmats = torch.cat(
            (axis_angle_to_rotation_matrix(pose[:, :3]).reshape(B, T, 1, 3, 3), rotmats[:, :, 1:, :, :]), dim=2)
        pose = pose.reshape(B, T, 72)
        pc = affine(data['human_points'].reshape(B, -1, 3), lidar_to_mocap_RT).reshape(B, T, 512, 3)
        #full_joints = self.smpl.get_full_joints(self.smpl(rotmats.reshape(-1, 24, 3, 3).cuda(), torch.zeros((B * T, 10)).cuda())).reshape(B, T, 24, 3).cpu().detach()
        full_joints = self.smpl.get_full_joints(self.smpl(rotmats.reshape(-1, 24, 3, 3), torch.zeros((B * T, 10)))).reshape(B, T, 24, 3).detach()

        data['pose'] = pose.reshape(T, 72)
        data['rotmats'] = rotmats.reshape(T, 24, 3, 3)
        data['human_points'] = pc.reshape(T, 512, 3)
        data['full_joints'] = full_joints.reshape(T, 24, 3)

    def __getitem__(self, index):
        if not hasattr(self, 'datas'):
            self.open_hdf5()

        item = {}
        item['index'] = index
        try:
            pose, betas, trans, human_points, points_num, full_joints, rotmats, lidar_to_mocap_RT, body_label = self.access_hdf5(index)
        except NotImplementedError as e:
            print(e)
            print(f'[ERROR]access_hdf5 error, index is {index}, hdf5 is {self.cfg.dataset_ids[self.acquire_hdf5_by_index(index)[0]]}')

        if self.cfg.ret_raw_pc:
            item['point_clouds'] = human_points.copy()

        if self.cfg.add_noice_pc:
            assert body_label is None, '目前只支持在无body_label的数据集上添加噪音'
            unique_pc = [np.unique(seg, axis=0) for seg in human_points]
            noice_pc = []
            body_label = []
            for e in unique_pc:
                numa = int((512 - e.shape[0]) * self.cfg.noice_pc_rate)
                numb = 512 - numa - e.shape[0]
                noice_pc.append(np.concatenate(
                    (e,
                     e[np.random.choice(np.arange(e.shape[0]), numb)],
                     (np.random.rand(numa, 3) - 0.5) * self.cfg.noice_pc_scale + e.mean(axis=0, keepdims=True),
                     ), axis=0))
                body_label.append(np.concatenate((np.ones(512 - numa), np.zeros(numa)), axis=0))
            human_points = np.stack(noice_pc)
            body_label = np.stack(body_label)

        if self.cfg.set_body_label_all_one:
            body_label = np.ones(human_points.shape[:2])

        if self.cfg.use_trans_to_normalize:
            points = human_points.copy() - trans[:, np.newaxis, :]
            #points = human_points.copy() - trans[7:8, np.newaxis, :]
            #points -= np.mean(points[7:8, :, :], axis=1)
        elif self.cfg.use_pc_w_raw_z:
            points = pc_normalize_w_raw_z(human_points.copy())
        else:
            points = pc_normalize(human_points.copy())

        noice_point_num = int(512 * self.cfg.replace_noice_pc_rate)
        if self.cfg.replace_noice_pc:
            # 将人体点云中一些点随机替换为噪音点
            if noice_point_num > 0:
                noice_choice = np.random.choice(np.arange(512), noice_point_num, replace=False)
                points[np.arange(16)[:, np.newaxis], noice_choice[np.newaxis, :]] = (np.random.rand(16, noice_point_num, 3) - 0.5) * np.array([0.8, 0.8, 1.2])
                if body_label is None:
                    body_label = np.ones(human_points.shape[:2])
                body_label[np.arange(16)[:, np.newaxis], noice_choice[np.newaxis, :]] = 0
            else:
                if body_label is None:
                    body_label = np.ones(human_points.shape[:2])

        if self.cfg.random_permutation:
            permuatation = np.random.permutation(512)
            points = points[np.arange(16)[:, np.newaxis], permuatation[np.newaxis, :]]
            body_label = body_label[np.arange(16)[:, np.newaxis], permuatation[np.newaxis, :]]

        item['human_points'] = torch.from_numpy(points).float()
        item['pose'] = torch.from_numpy(pose).float()

        if self.cfg.use_aug:
            points_num = points_num
            augmented_points = augment(points, points_num)
            item['human_points'] = torch.from_numpy(augmented_points).float()

        item['points_num'] = torch.from_numpy(points_num).int()
        item['betas'] = torch.from_numpy(betas).float()
        item['trans'] = torch.from_numpy(trans).float()

        if full_joints is not None:
            item['full_joints'] = torch.from_numpy(full_joints).float()
        if rotmats is not None:
            item['rotmats'] = torch.from_numpy(rotmats).float()
        if self.lidar_to_mocap_RT_flag:
            item['lidar_to_mocap_RT'] = torch.from_numpy(lidar_to_mocap_RT).float()
        if body_label is not None:
            item['body_label'] = body_label.astype(np.float64)


        if self.cfg.use_rot or self.cfg.use_straight:
            self.rot(item)

        return item

    def __len__(self):
        return self.length

def fix_dataset_seqlen(dataset_id):
    import h5py
    output_dataset_name = os.path.join('your_data_path', f'{dataset_id}_fix_dataset_seqlen.hdf5')
    dataset_name = os.path.join('your_data_path', f'{dataset_id}.hdf5')

    print(f'正在读取数据集：{dataset_name}')
    dataset = h5py.File(dataset_name, 'r')

    with h5py.File(output_dataset_name, 'w') as f:
        for k, v in dataset.items():
            print('原数据集', k, v.shape)
            if len(v.shape) >= 2 and v.shape[1] == 16:
                new_v = v[:].reshape(v.shape[0] * v.shape[1], *v.shape[2:])
                print('新数据集', k, new_v.shape)
                f.create_dataset(k, data=new_v)
    print(f'成功将数据集：{dataset_id}中的seqlen变为0！')
    print(f'新数据集保存在：{output_dataset_name}')


if __name__ == '__main__':
    from yacs.config import CfgNode
    #from OVis import ovis
    cfg = CfgNode.load_cfg(open(os.path.join(os.path.dirname(__file__), 'segv4_dn_v2dataset.yaml')))


    dataset = Lidarcapv2_Dataset(cfg.TrainDataset)
    dataset.update_cfg(dataset_ids=[51810, ], replace_noice_pc=True, replace_noice_pc_rate=0.2, use_trans_to_normalize=True)
    loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=8,shuffle=True)

    for x in tqdm(loader):
        #print(x['human_points'].shape)
        pass

    loader.dataset.update_cfg(drop_first_n=13)
    for x in tqdm(loader):
        #print(x['human_points'].shape)
        pass
    print()


if __name__ == '__main__':
    import socketio

    client = socketio.Client()
    client.connect('http://xxx.xx.xx.xx:xx')



    cfg = CfgNode.load_cfg(open(os.path.join(os.path.dirname(__file__), 'body_label.yaml')))
    dataset = Lidarcapv2_Dataset(cfg.TrainDataset)

    x = dataset.__getitem__(0)
    dataset.use_rot = True
    dataset.use_straight = True
    dataset.rot(x)
    import time

    for i in range(16):
        client.emit('add_pc', ('pc', x['human_points'][i].tolist()))
        client.emit('add_smpl_mesh', ('human_mesh', x['pose'][i].tolist(), None, None))
        time.sleep(0.1)