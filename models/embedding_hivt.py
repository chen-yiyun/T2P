"""
HiVT模型的嵌入层模块,包含单输入和多输入两种嵌入实现。

主要组件:
- SingleInputEmbedding_hivt: 单输入嵌入层,处理单一类型的输入特征
- MultipleInputEmbedding_hivt: 多输入嵌入层,可同时处理连续和离散特征
"""

from typing import List, Optional

import torch
import torch.nn as nn

from utils import init_weights


class SingleInputEmbedding_hivt(nn.Module):
    """单输入嵌入层
    
    将单一类型的输入特征映射到指定维度的嵌入空间。
    使用多层MLP进行特征变换,每层包含Linear、LayerNorm和ReLU。
    
    参数:
        in_channel (int): 输入特征维度
        out_channel (int): 输出嵌入维度
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int) -> None:
        super(SingleInputEmbedding_hivt, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class MultipleInputEmbedding_hivt(nn.Module):
    """多输入嵌入层
    
    可同时处理多个连续特征和离散特征的输入。
    对每个输入特征使用独立的MLP进行变换,然后聚合。
    
    参数:
        in_channels (List[int]): 各输入特征的维度列表
        out_channel (int): 输出嵌入维度
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int) -> None:
        super(MultipleInputEmbedding_hivt, self).__init__()
        # 为每个输入特征创建独立的MLP
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        # 聚合后的特征变换
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """前向传播
        
        参数:
            continuous_inputs: 连续特征列表
            categorical_inputs: 离散特征列表(可选)
            
        返回:
            聚合后的嵌入特征
        """
        # 处理连续特征
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)
        
        # 处理离散特征(如果有)
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
            
        return self.aggr_embed(output)
