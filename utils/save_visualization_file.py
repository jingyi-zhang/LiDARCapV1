# -*- coding: utf-8 -*-
# @Author  : Zhang.Jingyi

import torch
from ._init_path import *
import numpy as np
import pickle
import cv2 as cv


def save_ply(vertice, out_file):
    if type(vertice) == torch.Tensor:
        vertice = vertice.squeeze().cpu().detach().numpy()
    if vertice.ndim == 3:
        assert vertice.shape[0] == 1
        vertice = vertice.squeeze(0)
    model_file = 'lidarcap_fusion/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    with open(model_file, 'rb') as f:
        smpl_model = pickle.load(f, encoding='iso-8859-1')
        face_index = smpl_model['f'].astype(np.int64)
    face_1 = np.ones((face_index.shape[0], 1))
    face_1 *= 3
    face = np.hstack((face_1, face_index)).astype(int)
    with open(out_file, "wb") as zjy_f:
        np.savetxt(zjy_f, vertice, fmt='%f %f %f')
        np.savetxt(zjy_f, face, fmt='%d %d %d %d')
    ply_header = '''ply
format ascii 1.0
element vertex 6890
property float x
property float y
property float z
element face 13776
property list uchar int vertex_indices
end_header
    '''
    with open(out_file, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header)
        f.write(old)


def crop_image(image, box, outfile):
    images = cv.imread(image)
    a = int(box[-1].cpu().numpy())
    b = int(box[1].cpu().numpy())
    c = int(box[-2].cpu().numpy())
    d = int(box[0].cpu().numpy())
    cv.rectangle(images, (d, b), (c, a), (72, 118, 214), 4)
    cv.imwrite(outfile, images)
