"""
HiVT模型的全局交互模块,用于建模智能体间的交互关系。

主要组件:
- GlobalInteractor_hivt: 全局交互器,包含多层交互层
- GlobalInteractorLayer_hivt: 全局交互层,基于注意力机制实现智能体间的消息传递

主要功能:
- 提取智能体间的相对位置和方向特征
- 通过多头注意力机制建模智能体间的交互
- 生成多模态的全局交互特征
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from models import MultipleInputEmbedding_hivt
from models import SingleInputEmbedding_hivt
from utils import TemporalData
from utils import init_weights


class GlobalInteractor_hivt(nn.Module):
    """全局交互器
    
    参数:
        historical_steps (int): 历史步长
        embed_dim (int): 嵌入维度
        edge_dim (int): 边特征维度
        num_modes (int): 预测模态数
        num_heads (int): 注意力头数
        num_layers (int): 交互层数
        dropout (float): Dropout比例
        rotate (bool): 是否使用旋转变换
    """

    def __init__(
        self,
        historical_steps: int,
        embed_dim: int,
        edge_dim: int,
        num_modes: int = 6,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        rotate: bool = True,
    ) -> None:
        super(GlobalInteractor_hivt, self).__init__()
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes

        # 相对位置编码器
        if rotate:
            self.rel_embed = MultipleInputEmbedding_hivt(
                in_channels=[edge_dim, 2], out_channel=embed_dim
            )
        else:
            self.rel_embed = SingleInputEmbedding_hivt(
                in_channel=edge_dim, out_channel=embed_dim
            )
            
        # 多层交互层
        self.global_interactor_layers = nn.ModuleList(
            [
                GlobalInteractorLayer_hivt(
                    embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        
        # 输出层
        self.norm = nn.LayerNorm(embed_dim)
        self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        self.apply(init_weights)

    def forward(self, data: TemporalData, local_embed: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            data: 时序数据
            local_embed: 局部特征 [N, D]
            
        返回:
            全局交互特征 [F, N, D]
        """
        # 提取非填充的边
        edge_index, _ = subgraph(
            subset=~data["padding_mask"][:, self.historical_steps - 1],
            edge_index=data.edge_index,
        )
        
        # 计算相对位置
        rel_pos = (
            data["positions"][edge_index[0], self.historical_steps - 1]
            - data["positions"][edge_index[1], self.historical_steps - 1]
        )
        
        # 计算相对位置编码
        if data["rotate_mat"] is None:
            rel_embed = self.rel_embed(rel_pos)
        else:
            # 应用旋转变换
            rel_pos = torch.bmm(
                rel_pos.unsqueeze(-2), data["rotate_mat"][edge_index[1]]
            ).squeeze(-2)
            rel_theta = (
                data["rotate_angles"][edge_index[0]]
                - data["rotate_angles"][edge_index[1]]
            )
            rel_theta_cos = torch.cos(rel_theta).unsqueeze(-1)
            rel_theta_sin = torch.sin(rel_theta).unsqueeze(-1)
            rel_embed = self.rel_embed(
                [rel_pos, torch.cat((rel_theta_cos, rel_theta_sin), dim=-1)]
            )
            
        # 多层交互
        x = local_embed
        for layer in self.global_interactor_layers:
            x = layer(x, edge_index, rel_embed)
            
        # 生成多模态输出
        x = self.norm(x)  # [N, D]
        x = self.multihead_proj(x).view(-1, self.num_modes, self.embed_dim)  # [N, F, D]
        x = x.transpose(0, 1)  # [F, N, D]
        return x


class GlobalInteractorLayer_hivt(MessagePassing):
    """全局交互层
    
    基于图注意力机制实现智能体间的消息传递。
    
    参数:
        embed_dim (int): 嵌入维度
        num_heads (int): 注意力头数
        dropout (float): Dropout比例
    """

    def __init__(
        self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, **kwargs
    ) -> None:
        super(GlobalInteractorLayer_hivt, self).__init__(
            aggr="add", node_dim=0, **kwargs
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 注意力层参数
        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        
        # 更新层参数
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # 规范化和前馈层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_attr: torch.Tensor,
        size: Size = None,
    ) -> torch.Tensor:
        """前向传播"""
        x = x + self._mha_block(self.norm1(x), edge_index, edge_attr, size)
        x = x + self._ff_block(self.norm2(x))
        return x

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> torch.Tensor:
        """消息传递"""
        # 计算查询、键、值
        query = self.lin_q_node(x_i).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
        )
        key_node = self.lin_k_node(x_j).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
        )
        key_edge = self.lin_k_edge(edge_attr).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
        )
        value_node = self.lin_v_node(x_j).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
        )
        value_edge = self.lin_v_edge(edge_attr).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
        )
        
        # 计算注意力权重
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        
        # 加权聚合
        return (value_node + value_edge) * alpha.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """更新节点特征"""
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs + gate * (self.lin_self(x) - inputs)

    def _mha_block(
        self, x: torch.Tensor, edge_index: Adj, edge_attr: torch.Tensor, size: Size
    ) -> torch.Tensor:
        """多头注意力块"""
        x = self.out_proj(
            self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)
        )
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """前馈网络块"""
        return self.mlp(x)
