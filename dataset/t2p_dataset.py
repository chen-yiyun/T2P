from torch_geometric.data import Dataset
import torch
import numpy as np
import copy
import open3d as o3d
import glob

class T2PDataset(Dataset):
    """
    T2P数据集类,用于处理和加载轨迹预测数据
    """
    def __init__(self, dataset, mode=0, device='cuda', transform=None, input_time=None):
        """
        初始化函数
        Args:
            dataset: 数据集名称,目前支持'3dpw'
            mode: 0表示训练集,1表示验证集
            device: 计算设备
            transform: 数据增强变换
            input_time: 输入序列长度
        """
        self.dataset = dataset
        if dataset == "3dpw":
            self.num_person = 2  # 每个场景2个人
            if input_time != 10: raise Exception('Input time step other than 10 is not implemented yet')
            # 根据mode加载训练集或验证集
            if mode==0:
                self.data = glob.glob('D:/Downloads/T2P/T2P-main/preprocessed/3dpw_input10_v2/train/*.pt')
            elif mode==1:
                self.data = glob.glob('D:/Downloads/T2P/T2P-main/preprocessed/3dpw_input10_v2/val/*.pt')
        self.data = sorted(self.data)
        self.len_ = len(self.data)
        self.device = device
        self.dataset = dataset
        self.transform = transform
        self.input_time = input_time
        super(T2PDataset, self).__init__(transform=transform)

    def get(self, idx):
        """
        获取指定索引的数据样本
        Args:
            idx: 数据索引
        Returns:
            data: 包含输入序列和输出序列的数据字典
        """
        # 加载数据
        data = torch.load(self.data[idx])
        
        # 数据增强:随机旋转场景
        if self.transform:   
            idx_ = np.random.randint(0, 3)
            rot = [np.pi, np.pi/2, np.pi/4, np.pi*2]
            points = data.body_xyz.numpy().reshape(-1, 3)
            # 创建点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # 旋转点云
            pcd_EulerAngle = copy.deepcopy(pcd)
            R1 = pcd.get_rotation_matrix_from_xyz((0, rot[idx_], 0))
            pcd_EulerAngle.rotate(R1)  
            pcd_EulerAngle.paint_uniform_color([0, 0, 1])
            data['body_xyz'] = torch.tensor(np.asarray(pcd_EulerAngle.points).reshape(-1, 75, 45))

        # 分离输入和输出序列
        input_seq = data.body_xyz[:, :self.input_time, :]
        output_seq = data.body_xyz[:, self.input_time:, :]

        # 转换数据类型
        input_seq = torch.as_tensor(input_seq, dtype=torch.float32)
        output_seq = torch.as_tensor(output_seq, dtype=torch.float32)
        
        # 将最后一帧输入添加到输出序列开头
        last_input = input_seq[:, -1:, :]
        output_seq = torch.cat([last_input, output_seq], dim=1)
        
        # 重塑数据维度以匹配每个场景的人数
        data['input_seq'] = input_seq.reshape(input_seq.shape[0]//self.num_person, self.num_person, input_seq.shape[1], -1)
        data['output_seq'] = output_seq.reshape(output_seq.shape[0]//self.num_person, self.num_person, output_seq.shape[1], -1)
        
        # 将bos_mask移到CPU
        data.bos_mask = data.bos_mask.cpu()
        return data

    def len(self):
        """返回数据集长度"""
        return self.len_
