import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.geometry import rotation_matrix_to_axis_angle
from modules.smpl import SMPL
from scipy.spatial.transform import Rotation as R
from utils import path_util
from plyfile import PlyData, PlyElement
from skimage import io
import joblib
import json
import numpy as np
import torch


def get_sampled_pointcloud(idx):
    pointcloud_folder = 'lidarcap/labels/3d/sampled/{}'.format(
        idx)
    pointcloud_filenames = path_util.get_sorted_filenames_by_index(
        pointcloud_folder)
    return pointcloud_filenames


def get_segment(idx):
    pointcloud_folder = 'lidarcap/labels/3d/segment/{}'.format(
        idx)
    pointcloud_filenames = path_util.get_sorted_filenames_by_index(
        pointcloud_folder)
    return pointcloud_filenames


def get_images(idx):
    image_folder = 'lidarcap/images/{}'.format(idx)
    image_filenames = path_util.get_sorted_filenames_by_index(image_folder)
    image_indexes = np.load(os.path.join(image_folder, 'image_indexes.npy'))
    start = image_indexes[0]
    rectified_start = start / 30 * 29.83
    diff = start - rectified_start
    image_indexes = image_indexes - diff
    image_indexes = np.around(image_indexes).astype(int)
    image_indexes -= 1
    image_filenames = np.array(image_filenames)[image_indexes]
    return image_filenames


def get_vibe_images(idx):
    image_folder = 'VIBE/{}_vibe_output'.format(idx)
    image_filenames = path_util.get_sorted_filenames_by_index(image_folder)
    return image_filenames


def get_hmr_images(idx):
    image_folder = 'hmr/{}'.format(idx)
    image_filenames = path_util.get_sorted_filenames_by_index(image_folder)
    return image_filenames


def get_hmr_poses(idx):
    filename = 'hmr/{}_hmr.npy'.format(idx)
    hmr_poses = np.load(filename)
    if idx == 24 or idx == 29:
        hmr_poses = hmr_poses[:-1]
    hmr_poses = hmr_poses.squeeze()
    return hmr_poses


def get_vibe_poses(idx):
    filename = 'VIBE/{}_vibe_output.pkl'.format(idx)
    vibe_result = joblib.load(filename)

    vibe_result = vibe_result[list(vibe_result.keys())[0]]
    vibe_poses = vibe_result['pose']

    lidar_to_camera_RT = np.array([-0.0043368991524, -0.99998911867, -0.0017186757713, 0.016471385748, -0.0052925495236, 0.0017416212982, -
                                   0.99998447772, 0.080050847871, 0.99997658984, -0.0043277356572, -0.0053000451695, -0.049279053295, 0, 0, 0, 1]).reshape(4, 4)

    vibe_poses = camera_poses_to_lidar(vibe_poses, lidar_to_camera_RT)
    return vibe_poses


def get_gt_poses(idx):
    pose_folder = 'lidarcap/labels/3d/pose/{}'.format(idx)
    gt_poses = []
    for pose_filename in filter(lambda x: x.endswith('.json'), path_util.get_sorted_filenames_by_index(pose_folder)):
        with open(pose_filename) as f:
            content = json.load(f)
            gt_pose = np.array(content['pose'], dtype=np.float32)
            gt_poses.append(gt_pose)
    gt_poses = np.stack(gt_poses)
    return gt_poses


def get_gt_trans(idx):
    pose_folder = 'lidarcap/labels/3d/pose/{}'.format(idx)
    gt_trans = []
    for pose_filename in filter(lambda x: x.endswith('.json'), path_util.get_sorted_filenames_by_index(pose_folder)):
        with open(pose_filename) as f:
            content = json.load(f)
            gt_pose = np.array(content['trans'], dtype=np.float32)
            gt_trans.append(gt_pose)
    gt_trans = np.stack(gt_trans)
    return gt_trans


def images_to_videos(video_path, images_dir):
    assert os.path.isabs(video_path) and os.path.isfile(video_path)
    assert os.path.isabs(images_dir) and os.path.isdir(images_dir)
    os.system('ffmpeg -r 10 -y -threads 16 -i {}/%06d.png -profile:v baseline -level 3.0 -c:v libx264 -pix_fmt yuv420p -an -v error {}'.format(images_dir, video_path))


def get_pred_poses(name, idx):
    filename = 'exp/{}/lidarcap_{}.npy'.format(name, idx)
    pred_rotmats = np.load(filename).reshape(-1, 24, 3, 3)
    pred_poses = []
    for pred_rotmat in pred_rotmats:
        pred_poses.append(rotation_matrix_to_axis_angle(
            torch.from_numpy(pred_rotmat)).numpy().reshape((72, )))
    pred_poses = np.stack(pred_poses)
    return pred_poses


def read_point_cloud(filename):
    """ read XYZ point cloud from filename PLY file """
    ply_data = PlyData.read(filename)['vertex'].data
    points = np.array([[x, y, z] for x, y, z in ply_data])
    return points


def camera_to_pixel(X, intrinsic_matrix, distortion_coefficients):
    # focal length
    f = np.array([intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]])
    # center principal point
    c = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])
    k = np.array([distortion_coefficients[0],
                 distortion_coefficients[1], distortion_coefficients[4]])
    p = np.array([distortion_coefficients[2], distortion_coefficients[3]])
    XX = X[..., :2] / X[..., 2:]
    r2 = np.sum(XX[..., :2]**2, axis=-1, keepdims=True)

    radial = 1 + np.sum(k * np.concatenate((r2, r2**2, r2**3),
                        axis=-1), axis=-1, keepdims=True)

    tan = 2 * np.sum(p * XX[..., ::-1], axis=-1, keepdims=True)
    XXX = XX * (radial + tan) + r2 * p[..., ::-1]
    return f * XXX + c


def affine(X, matrix):
    n = X.shape[0]
    res = np.concatenate((X, np.ones((n, 1))), axis=-1).T
    res = np.dot(matrix, res).T
    return res[..., :-1]


def lidar_to_camera(X, extrinsic_matrix):
    return affine(X, extrinsic_matrix)


def camera_poses_to_lidar(poses, extrinsic_matrix):
    lidar_to_camera_R = extrinsic_matrix[:3, :3]
    camera_to_lidar_R = np.linalg.inv(lidar_to_camera_R)

    for pred_pose in poses:
        pred_pose[:3] = (R.from_matrix(camera_to_lidar_R) *
                         R.from_rotvec(pred_pose[:3])).as_rotvec()
    return poses


def camera_to_lidar(X, extrinsic_matrix):
    return affine(X, np.linalg.inv(extrinsic_matrix))


def get_bbox(segment_filename):
    extrinsic_matrix = np.array([-0.0043368991524, -0.99998911867, -0.0017186757713, 0.016471385748, -0.0052925495236, 0.0017416212982, -
                                 0.99998447772, 0.080050847871, 0.99997658984, -0.0043277356572, -0.0053000451695, -0.049279053295, 0, 0, 0, 1]).reshape(4, 4)
    intrinsic_matrix = np.array([9.5632709662202160e+02, 0., 9.6209910493679433e+02,
                                0., 9.5687763573729683e+02, 5.9026610775785059e+02, 0., 0., 1.]).reshape(3, 3)
    distortion_coefficients = np.array([-6.1100617222502205e-03, 3.0647823796371827e-02, -
                                        3.3304524444662654e-04, -4.4038460096976607e-04, -2.5974982760794661e-02])

    lidar_points = read_point_cloud(segment_filename)
    pixel_points = camera_to_pixel(lidar_to_camera(
        lidar_points, extrinsic_matrix), intrinsic_matrix, distortion_coefficients)
    pixel_points = pixel_points.astype(int)
    pixel_points = pixel_points[pixel_points[:, 0] > -1]
    pixel_points = pixel_points[pixel_points[:, 0] < 1920]
    pixel_points = pixel_points[pixel_points[:, 1] > -1]
    pixel_points = pixel_points[pixel_points[:, 1] < 1080]
    x_min = pixel_points[:, 0].min()
    y_min = pixel_points[:, 1].min()
    x_max = pixel_points[:, 0].max()
    y_max = pixel_points[:, 1].max()
    x_length = x_max - x_min
    y_length = y_max - y_min
    length = max(x_length + 40, y_length + 40)
    length_half = math.floor(length // 2)
    x_center = math.floor((x_max + x_min) // 2)
    y_center = math.floor((y_max + y_min) // 2)
    return np.array([x_center - length_half, y_center - length_half, length, length])
    # return np.array([x_min - 10, y_min - 10, max(x_length + 20, y_length + 20), max(x_length + 20, y_length + 20)])


def read_image(image_filename):
    return io.imread(image_filename)


# use pcl to write ply has bug
def save_point_cloud(filename, points):
    points = [(points[i, 0], points[i, 1], points[i, 2])
              for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=False).write(filename)


def write_image(image_filename, image):
    return io.imsave(image_filename, image)


def crop_image(image, bbox):
    # bbox [x, y, w, h]
    x, y, w, h = bbox
    return image[y:y + h, x:x + w]


def poses_to_vertices(poses, trans=None):
    poses = poses.astype(np.float32)
    vertices = np.zeros((0, 6890, 3))

    n = len(poses)
    smpl = SMPL().cuda()
    batch_size = 128
    n_batch = (n + batch_size - 1) // batch_size

    for i in range(n_batch):
        lb = i * batch_size
        ub = (i + 1) * batch_size

        cur_n = min(ub - lb, n - lb)
        cur_vertices = smpl(torch.from_numpy(
            poses[lb:ub]).cuda(), torch.zeros((cur_n, 10)).cuda())
        vertices = np.concatenate((vertices, cur_vertices.cpu().numpy()))
    if trans is not None:
        trans = trans.astype(np.float32)
        vertices += np.expand_dims(trans, 1)
    return vertices


def poses_to_joints(poses):
    poses = poses.astype(np.float32)
    joints = np.zeros((0, 24, 3))

    n = len(poses)
    smpl = SMPL().cuda()
    batch_size = 128
    n_batch = (n + batch_size - 1) // batch_size

    for i in range(n_batch):
        lb = i * batch_size
        ub = (i + 1) * batch_size

        cur_n = min(ub - lb, n - lb)
        cur_vertices = smpl(torch.from_numpy(
            poses[lb:ub]).cuda(), torch.zeros((cur_n, 10)).cuda())
        cur_joints = smpl.get_full_joints(cur_vertices)
        joints = np.concatenate((joints, cur_joints.cpu().numpy()))
    return joints


if __name__ == '__main__':
    print(get_images(7)[:10])
