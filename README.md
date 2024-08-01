# Download LiDARHuman51M dataset
- Download the dataset and weight file from the link: http://www.lidarhumanmotion.net/lidarcap/.
- Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`,`J_regressor_extra.npy` from link: https://smpl.is.tue.mpg.de/
and set path in `utils/config.py`.

# TRAIN 
### 1. Modify the info
- Modify `segv5_dn_v2dataset.yaml`to set `dataset_path` to the path where your dataset is located.
- Update the relevant information for `wandb` in `tools/common.py`.
### 2. Build Environment
```
conda create -n lidar_human python=3.7
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"（或者下载github然后pip install pointnet2_ops_lib/.）
pip install wandb
pip install h5py
pip install tqdm
pip install scipy
pip install opencv-python
pip install pyransac3d
pip install yacs
pip install plyfile
pip install scikit-image
pip install joblib
pip install chumpy
```

### 3. Train 
```
python train.py --threads x --gpu x --config segv5_dn_v2dataset
```


### Citation
```@article{ZHANG2024110848,
title = {LiDARCapV2: 3D human pose estimation with human-object interaction from LiDAR point clouds},
journal = {Pattern Recognition},
pages = {110848},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.110848},
url = {https://www.sciencedirect.com/science/article/pii/S0031320324005995},
author = {Jingyi Zhang and Qihong Mao and Siqi Shen and Chenglu Wen and Lan Xu and Cheng Wang},}```