import socketio
client = socketio.Client()
client.connect('http://172.18.69.63:5666')

import h5py
import numpy as np

dataset_name = '/SAMSUMG8T/mqh/lidarcapv2/dataset/50301.hdf5'
dataset_name = '/SAMSUMG8T/mqh/lidarcapv2/dataset/mask_lidarcap_train2.hdf5'
with h5py.File(dataset_name, 'r') as d:
    import time
    for i in range(0, len(d['pose']) * 16, 1):
        client.emit('add_smpl_mesh', ('human_pc', d['pose'][i].tolist(), None, d['trans'][i].tolist()))
        #client.emit('add_pc', ('pc', d['point_clouds'][i].tolist()))
        client.emit('add_pc', ('pc', d['masked_point_clouds'][i].tolist()))
        client.emit('add_pc', ('pc2', (d['masked_point_clouds'][i][~d['body_label'][i]] + np.array([[0, 1, 0]])).tolist()))
        time.sleep(0.1)