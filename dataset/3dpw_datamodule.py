"""
JRDB数据集的数据模块类,继承自LightningDataModule
"""
# 导入所需的库
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict
from functools import partial
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader
from dataset.t2p_dataset import T2PDataset


# JRDB数据集的数据模块类,继承自LightningDataModule
class jrdb_DataModule(LightningDataModule):
    def __init__(
        self,
        train_args: Dict,  # 训练参数字典
        val_args: Dict,    # 验证参数字典 
        test_args: Dict,   # 测试参数字典
        shuffle: bool = True,  # 是否打乱数据
        num_workers: int = 8,  # 数据加载的工作进程数
        pin_memory: bool = True,  # 是否将数据固定在内存中
    ):
        super(jrdb_DataModule, self).__init__()
        # 初始化参数字典
        self.train_args, self.val_args, self.test_args = {}, {}, {}
        # 更新各个阶段的参数
        for _arg in train_args: self.train_args.update(_arg)
        for _arg in val_args: self.val_args.update(_arg)
        for _arg in test_args: self.test_args.update(_arg)

        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    # 设置数据集
    def setup(self, stage: Optional[str] = None) -> None:
        # 创建训练、验证和测试数据集实例
        self.train_dataset = T2PDataset(self.train_args['dataset'], mode=0, input_time=self.train_args['input_time'])
        self.val_dataset = T2PDataset(self.val_args['dataset'], mode=1, input_time=self.val_args['input_time'])
        self.test_dataset = T2PDataset(self.test_args['dataset'], mode=1, input_time=self.test_args['input_time'])

    # 自定义数据收集函数(已注释)
    # def custom_collate_fn(batch):
    #     # Move tensors to the default device
    #     import pdb;pdb.set_trace()
    #     batch = [(item[0].to('cuda'), item[1].to('cuda')) for item in batch]
    #     return default_collate(batch)
    
    # 返回训练数据加载器
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_args['bs'],
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    # 返回验证数据加载器
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_args['bs'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    # 返回测试数据加载器
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_args['bs'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
