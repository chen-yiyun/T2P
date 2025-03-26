"""
3DPW数据集预处理
"""

# 导入所需的库
import os
import os.path as osp
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
import torch

from multiprocessing import Process
from multiprocessing import Pool
from itertools import repeat

from scipy.spatial.distance import cdist

from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data.dataset import files_exist
from tqdm import tqdm
import pickle as pkl

import random

import sys

# 添加项目根目录到系统路径
sys.path.append("D:/Downloads/T2P/T2P-main")
from utils import TemporalData
from debug_util import *

# 定义常量
PED_CLASS = {}
VEH_CLASS = {}
SPLIT_NAME = {"3dpw": {"train": "train", "val": "val", "test": "val"}}
RAW_FILE_NAMES_3DPW = {
    "train": "D:/Downloads/T2P/T2P-main/data/poseData.pkl",
    "val": "D:/Downloads/T2P/T2P-main/data/poseData.pkl",
    "test": "D:/Downloads/T2P/T2P-main/data/poseData.pkl",
}


# 定义绕X轴旋转矩阵
def Rx(theta):
    return np.matrix(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


# 定义绕Y轴旋转矩阵
def Ry(theta):
    return np.matrix(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


# 定义绕Z轴旋转矩阵
def Rz(theta):
    return np.matrix(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


# 定义旋转角度和旋转矩阵
rotate_x = np.radians(-90)
R = Rx(rotate_x)


# 定义3DPW数据集类
class _3DPW(Dataset):

    def __init__(
        self,
        split: str,
        root: str,
        process_dir: str,
        transform: Optional[Callable] = None,
        local_radius: float = 50,
        process: bool = False,
        spec_args: Dict = None,
    ) -> None:
        self._split = split
        self._local_radius = local_radius
        # 设置特定参数
        for k, v in spec_args.items():
            self.__setattr__(k, v)

        self._directory = SPLIT_NAME[self.dataset][split]
        self.process_dir = process_dir
        # 根据不同的数据集划分读取相应的数据
        with open(RAW_FILE_NAMES_3DPW[split], "rb") as input:
            if split == "train":
                # 读取训练集数据
                oridata = pkl.load(input)["{}".format("train")]
                # 对原始数据进行轴向滚动,生成额外的训练样本
                p_oridata = np.roll(np.array(oridata), 1, axis=1)
                # 将原始数据和滚动后的数据拼接在一起
                self._raw_file = np.concatenate((oridata, p_oridata), axis=0)
            elif split == "val":
                # 读取验证集数据
                oridata = pkl.load(input)["{}".format("valid")]
                # 将数据堆叠成数组
                self._raw_file = np.stack(oridata)
            elif split == "test":
                # 读取测试集数据
                oridata = pkl.load(input)["{}".format("test")]
                # 将数据堆叠成数组
                self._raw_file = np.stack(oridata)

        # 处理原始数据的形状
        # 获取原始数据的维度信息
        # B: batch size 序列的数量
        # N: number of agents 智能体数量
        # T: time steps 时间步长
        # JXYZ: 关节坐标维度(x,y,z坐标值)
        B, N, T, JXYZ = self._raw_file.shape

        # 重塑数据维度:
        # 1. 将JXYZ除以3得到每个关节的3D坐标 (JXYZ//3个关节,每个关节3个坐标)
        # 2. 重排坐标轴顺序为[z,x,y],以匹配目标坐标系
        # 最终结果维度为: [B序列数, N智能体数, T时间步, JXYZ//3关节数, 3坐标值zxy]
        # 例如: 若B=100, N=5, T=20, JXYZ=60, 则结果维度为[100, 5, 20, 20, 3]
        self._raw_file = self._raw_file.reshape((B, N, T, JXYZ // 3, 3))
        self._raw_file = self._raw_file[..., [2, 0, 1]]

        # 生成文件名列表,用索引作为文件名
        self._raw_file_names = [i for i in range(self._raw_file.shape[0])]

        # 设置处理后的文件名和路径
        self._processed_file_names = [
            str(f) + ".pt" for f in range(len(self._raw_file))
        ]
        self._processed_paths = [
            os.path.join(self.processed_dir, f) for f in self._processed_file_names
        ]
        super(_3DPW, self).__init__(root, transform=transform)

    def _download(self):
        return

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.process_dir, self._directory)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def _process(self):
        # 如果已经处理过,直接返回
        if files_exist(self.processed_paths):  # pragma: no cover
            print("Found processed files")
            return

        print("Processing...", file=sys.stderr)

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        print("Done!", file=sys.stderr)

    def process(self) -> None:
        # 多进程处理数据
        if self.n_jobs > 1:
            raw_file_list = []
            total_num = len(self._raw_file_names)
            num_per_proc = int(np.ceil(total_num / self.n_jobs))
            for proc_id in range(self.n_jobs):
                start = proc_id * num_per_proc
                end = min((proc_id + 1) * num_per_proc, total_num)
                raw_file_list.append(self._raw_file_names[start:end])

            procs = []
            for proc_id in range(self.n_jobs):
                process = Process(
                    target=self.process_3dpw,
                    args=(raw_file_list[proc_id], self._local_radius),
                )
                process.daemon = True
                process.start()
                procs.append(process)

            for proc in procs:
                proc.join()

        else:
            self.process_3dpw(self._raw_file_names, self._local_radius)

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        data = torch.load(self.processed_paths[idx])
        if self._split == "train":
            data = self.augment(data)
        return data

    def augment(self, data):
        # 数据增强:随机翻转
        if self.random_flip:
            if random.choice([0, 1]):
                data.x = data.x * torch.tensor([-1, 1])
                data.y = data.y * torch.tensor([-1, 1])
                data.positions = data.positions * torch.tensor([-1, 1])
                theta_x = torch.cos(data.theta)
                theta_y = torch.sin(data.theta)
                data.theta = torch.atan2(theta_y, -1 * theta_x)
                angle_x = torch.cos(data.rotate_angles)
                angle_y = torch.sin(data.rotate_angles)
                data.rotate_angles = torch.atan2(angle_y, -1 * angle_x)
                lane_angle_x = torch.cos(data.lane_rotate_angles)
                lane_angle_y = torch.sin(data.lane_rotate_angles)
                data.lane_rotate_angles = torch.atan2(lane_angle_y, -1 * lane_angle_x)
                data.lane_positions = data.lane_positions * torch.tensor([-1, 1])
                data.lane_vectors = data.lane_vectors * torch.tensor([-1, 1])
                data.lane_actor_vectors = data.lane_actor_vectors * torch.tensor(
                    [-1, 1]
                )
            if random.choice([0, 1]):
                data.x = data.x * torch.tensor([1, -1])
                data.y = data.y * torch.tensor([1, -1])
                data.positions = data.positions * torch.tensor([1, -1])
                theta_x = torch.cos(data.theta)
                theta_y = torch.sin(data.theta)
                data.theta = torch.atan2(-1 * theta_y, theta_x)
                angle_x = torch.cos(data.rotate_angles)
                angle_y = torch.sin(data.rotate_angles)
                data.rotate_angles = torch.atan2(-1 * angle_y, angle_x)
                lane_angle_x = torch.cos(data.lane_rotate_angles)
                lane_angle_y = torch.sin(data.lane_rotate_angles)
                data.lane_rotate_angles = torch.atan2(-1 * lane_angle_y, lane_angle_x)
                data.lane_positions = data.lane_positions * torch.tensor([1, -1])
                data.lane_vectors = data.lane_vectors * torch.tensor([1, -1])
                data.lane_actor_vectors = data.lane_actor_vectors * torch.tensor(
                    [1, -1]
                )

        return data

    def process_3dpw(self, tokens: str, radius: float) -> Dict:
        """
        处理3DPW数据集的主要步骤:
        1. 读取原始数据并转换为张量格式
        2. 分离输入轨迹和输出轨迹
        3. 计算相对位移
        4. 创建掩码矩阵
        5. 计算和应用旋转变换
        6. 整理并保存处理后的数据
        """
        # 遍历每个数据样本
        for token in tqdm(tokens):
            # 1. 读取原始数据
            N, T, _, _ = self._raw_file[token].shape
            seq_data = self._raw_file[token]
            body_xyz = torch.tensor(seq_data, dtype=torch.float)

            # 2. 分离输入输出轨迹
            input_traj, output_traj = torch.tensor(
                seq_data[:, : self.ref_time, 0], dtype=torch.float
            ), torch.tensor(
                seq_data[:, self.ref_time :, 0], dtype=torch.float
            )  # todo：Only hip joint info(只有髋关节信息)

            edge_index = (
                torch.LongTensor(list(permutations(range(N), 2))).t().contiguous()
            )

            # 3. 计算相对位移
            x = torch.cat((input_traj, output_traj), dim=1)
            positions = x.clone()
            x[:, self.ref_time :] = x[:, self.ref_time :] - x[
                :, self.ref_time - 1
            ].unsqueeze(-2)
            x[:, 1 : self.ref_time] = (
                x[:, 1 : self.ref_time] - x[:, : self.ref_time - 1]
            )
            x[:, 0] = torch.zeros(N, 3)
            y = x[:, self.ref_time :]

            # 4. 创建掩码
            padding_mask = torch.zeros(N, T, dtype=torch.bool)
            bos_mask = torch.zeros(N, self.ref_time, dtype=torch.bool).cuda()
            bos_mask[:, 0] = ~padding_mask[:, 0]
            bos_mask[:, 1 : self.ref_time] = (
                padding_mask[:, : self.ref_time - 1]
                & ~padding_mask[:, 1 : self.ref_time]
            )

            # 5. 计算旋转角度和应用旋转
            rotate_angles = torch.zeros(N, dtype=torch.float)
            for actor_id in range(N):
                heading_vector = (
                    x[actor_id, self.ref_time] - x[actor_id, self.ref_time - 1]
                )
                rotate_angles[actor_id] = torch.atan2(
                    heading_vector[1], heading_vector[0]
                )

            if self.rotate:
                rotate_mat = torch.empty(N, 3, 3)
                sin_vals = torch.sin(rotate_angles)
                cos_vals = torch.cos(rotate_angles)
                rotate_mat[:, 0, 0] = cos_vals
                rotate_mat[:, 0, 1] = -sin_vals
                rotate_mat[:, 0, 2] = 0
                rotate_mat[:, 1, 0] = sin_vals
                rotate_mat[:, 1, 1] = cos_vals
                rotate_mat[:, 1, 2] = 0
                rotate_mat[:, 2, 0] = 0
                rotate_mat[:, 2, 1] = 0
                rotate_mat[:, 2, 2] = 1
                if y is not None:
                    y = torch.bmm(y, rotate_mat)
            else:
                rotate_mat = None

            # 6. 整理并保存数据
            # 将处理后的数据整理成字典,包含以下字段:
            # body_xyz: 人体关节点的原始3D坐标序列 [N, T, J*3]
            # x: 参考帧之前的相对位移序列 [N, ref_time, 3]
            # positions: 所有帧的绝对位置序列 [N, T, 3]
            # rotate_angles: 每个actor的旋转角度 [N]
            # padding_mask: 时序数据的padding掩码 [N, T]
            # edge_index: 图结构的边连接关系 [2, num_edges]
            # bos_mask: 序列起始位置的掩码 [N, ref_time]
            # y: 参考帧之后的相对位移序列 [N, T-ref_time, 3]
            # num_nodes: 场景中actor的数量 N
            # rotate_mat: 每个actor的旋转矩阵 [N, 3, 3]
            processed = {
                "body_xyz": body_xyz,
                "x": x[:, : self.ref_time],
                "positions": positions,
                "rotate_angles": rotate_angles,
                "padding_mask": padding_mask,
                "edge_index": edge_index,
                "bos_mask": bos_mask,
                "y": y,
                "num_nodes": N,
                "rotate_mat": rotate_mat,
            }
            # 将字典转换为TemporalData对象并保存为pt文件
            data = TemporalData(**processed)
            torch.save(data, os.path.join(self.processed_dir, str(token) + ".pt"))
        return


# 计算四元数的偏航角
def quaternion_yaw(q: Quaternion) -> float:
    """
    从四元数计算偏航角。
    注意这只适用于表示激光雷达或全局坐标系中的盒子的四元数。
    不适用于相机坐标系中的盒子。
    :param q: 目标四元数
    :return: 偏航角(弧度)
    """

    # 投影到xy平面
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # 使用arctan计算偏航角
    yaw = np.arctan2(v[1], v[0])

    return yaw


# 修正偏航角
def correct_yaw(yaw: float) -> float:
    """
    nuScenes地图在y轴上被翻转,所以我们需要
    在旋转航向角时加上pi。
    :param yaw: 用于旋转图像的偏航角
    :return: 修正后的偏航角
    """
    if yaw <= 0:
        yaw = -np.pi - yaw
    else:
        yaw = np.pi - yaw

    return yaw


# 标准化角度到[-pi, pi]区间
def normalize_angle(angle):
    if angle < -torch.pi:
        angle += 2 * torch.pi
    elif angle > torch.pi:
        angle -= 2 * torch.pi
    return angle


# 主程序入口
if __name__ == "__main__":
    # 设置参数
    # 设置特定参数
    spec_args = {
        "dataset": "3dpw",  # 数据集名称
        "n_jobs": 0,  # 处理数据的进程数,0表示单进程
        "t_h": 2,  # 历史轨迹长度(秒)
        "t_f": 6,  # 预测轨迹长度(秒)
        "res": 2,  # 轨迹采样分辨率(Hz)
        "ref_time": 10,  # 参考时间长度(秒)
        "lseg_len": 10,  # 轨迹分段长度
        "lseg_angle_thres": 30,  # 轨迹分段角度阈值(度)
        "lseg_dist_thres": 2.5,  # 轨迹分段距离阈值(米)
        "random_flip": True,  # 是否随机翻转数据增强
        "rotate": True,  # 是否进行旋转数据增强
    }
    # 处理训练集
    A1D = _3DPW(
        "train",
        process_dir="D:/Downloads/T2P/T2P-main/preprocessed/3dpw_input10_v2",
        root="data/",
        process=True,
        spec_args=spec_args,
    )
    # 处理验证集
    A1D = _3DPW(
        "val",
        process_dir="D:/Downloads/T2P/T2P-main/preprocessed/3dpw_input10_v2",
        root="data/",
        process=True,
        spec_args=spec_args,
    )
    # 处理测试集
    A1D = _3DPW(
        "test",
        process_dir="D:/Downloads/T2P/T2P-main/preprocessed/3dpw_input10_v2",
        root="data/",
        process=True,
        spec_args=spec_args,
    )

    from torch_geometric.data.batch import Batch
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm
