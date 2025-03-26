import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # 定义数据集路径
# data_dir = 'data/Mocap_UMPM'
#
# # 示例：加载并可视化test_3_75_mocap_umpm.npy
# data = np.load(os.path.join(data_dir, 'test_3_75_mocap_umpm.npy'))
# print(data.shape) # (3000, 3, 75, 45)

data_dir = 'data/'
# # 示例：加载并可视化poseData.pkl
data = np.load(os.path.join(data_dir, 'poseData.pkl'))
print(data.shape)
