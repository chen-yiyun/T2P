"""
HiVT模型的解码器模块,包含GRU和MLP两种解码器实现。

主要组件:
- GRUDecoder_hivt: 基于GRU的解码器,用于生成轨迹预测
- MLPDecoder_hivt: 基于MLP的解码器,用于生成轨迹预测

两种解码器都支持:
- 多模态预测(num_modes)
- 不确定性估计(uncertain)
- 位置和尺度预测
"""

from typing import Tuple

import torch
import torch.nn as nn 
import torch.nn.functional as F

from utils import init_weights


class GRUDecoder_hivt(nn.Module):
    """基于GRU的轨迹解码器
    
    参数:
        local_channels (int): 局部特征维度
        global_channels (int): 全局特征维度  
        future_steps (int): 预测步长
        num_modes (int): 预测模态数
        uncertain (bool): 是否预测不确定性
        min_scale (float): 最小尺度
    """

    def __init__(self,
                 local_channels: int,
                 global_channels: int, 
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(GRUDecoder_hivt, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        # GRU层,用于序列生成
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
                          
        # 位置预测层
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2))
            
        # 不确定性预测层(可选)
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 2))
                
        # 模态概率预测层
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        参数:
            local_embed: 局部特征 [N, D]
            global_embed: 全局特征 [F, N, D]
            
        返回:
            预测轨迹 [F, N, H, 2/4]
            模态概率 [N, F]
        """
        # 计算模态概率
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
                                
        # 准备输入序列
        global_embed = global_embed.reshape(-1, self.input_size)  # [F x N, D]
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.repeat(self.num_modes, 1).unsqueeze(0)  # [1, F x N, D]
        
        # GRU生成序列
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)  # [F x N, H, D]
        
        # 预测位置
        loc = self.loc(out)  # [F x N, H, 2]
        
        if self.uncertain:
            # 预测不确定性
            scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
            return torch.cat((loc, scale),
                             dim=-1).view(self.num_modes, -1, self.future_steps, 4), pi  # [F, N, H, 4], [N, F]
        else:
            return loc.view(self.num_modes, -1, self.future_steps, 2), pi  # [F, N, H, 2], [N, F]


class MLPDecoder_hivt(nn.Module):
    """基于MLP的轨迹解码器
    
    参数:
        local_channels (int): 局部特征维度
        global_channels (int): 全局特征维度
        future_steps (int): 预测步长
        num_modes (int): 预测模态数
        uncertain (bool): 是否预测不确定性
        min_scale (float): 最小尺度
    """

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(MLPDecoder_hivt, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        # 特征聚合层
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU())
            
        # 位置预测层
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.future_steps * 3))
            
        # 不确定性预测层(可选)
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.future_steps * 3))
                
        # 模态概率预测层
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        参数:
            local_embed: 局部特征 [N, D]
            global_embed: 全局特征 [F, N, D]
            
        返回:
            预测轨迹 [F, N, H, 3/6]
            模态概率 [N, F]
            特征 [F, N, D]
        """
        # 计算模态概率
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
                                
        # 特征聚合
        out = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))
        
        # 预测位置
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 3)  # [F, N, H, 3]
        
        if self.uncertain:
            # 预测不确定性
            scale = F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 3) + 1.0
            scale = scale + self.min_scale  # [F, N, H, 3]
            return torch.cat((loc, scale), dim=-1), pi, out  # [F, N, H, 4], [N, F]
        else:
            return loc, pi, out  # [F, N, H, 3], [N, F]
