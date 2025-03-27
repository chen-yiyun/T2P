"""
这个文件实现了一个基于HIVT架构的局部编码器。主要包含以下几个类:

LocalEncoder_hivt: 局部编码器类,包含:
- 边缘距离过滤
- 智能体-智能体交互编码器(AAEncoder)
- 时序编码器(TemporalEncoder)
- 特征重塑层

AAEncoder_hivt: 智能体-智能体交互编码器,基于MessagePassing:
- 多头注意力机制处理节点间交互
- 位置编码和运动特征融合
- 残差连接和LayerNorm

TemporalEncoder_hivt: 时序编码器:
- Transformer编码器处理时序信息
- 位置编码和CLS token
- 掩码机制处理填充

TemporalEncoderLayer_hivt: 时序编码器层:
- 多头自注意力
- 前馈网络
- 残差连接和LayerNorm
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from models import MultipleInputEmbedding_hivt
from models import SingleInputEmbedding_hivt
from utils import DistanceDropEdge
from utils import TemporalData
from utils import init_weights


class LocalEncoder_hivt(nn.Module):
    """局部编码器,处理历史轨迹数据
    
    Args:
        reshape_dim: 重塑维度
        historical_steps: 历史步数
        node_dim: 节点特征维度
        edge_dim: 边特征维度
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        dropout: Dropout率
        num_temporal_layers: 时序编码器层数
        local_radius: 局部半径阈值
        parallel: 是否并行处理
        enc_feat_dim: 编码特征维度
    """

    def __init__(self,
                 reshape_dim: int,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 parallel: bool = False,
                 enc_feat_dim: int = None) -> None:
        super(LocalEncoder_hivt, self).__init__()
        self.historical_steps = historical_steps
        self.parallel = parallel
        self.reshape_dim = reshape_dim
        self.drop_edge = DistanceDropEdge(local_radius)
        self.aa_encoder = AAEncoder_hivt(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    parallel=parallel)
        self.temporal_encoder = TemporalEncoder_hivt(historical_steps=historical_steps,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers)
        # 特征重塑层
        self.reshape_enc = nn.Sequential(nn.Linear(reshape_dim, int((historical_steps+reshape_dim)/2)),
                                         nn.Linear(int((historical_steps+reshape_dim)/2), historical_steps)
                                        )
        self.reshape_enc_2_ = False
        if enc_feat_dim is not None and enc_feat_dim != embed_dim:
            self.reshape_enc_2_ = True
            self.reshape_enc_2 = nn.Sequential(nn.Linear(enc_feat_dim, embed_dim),
                                            )

    def forward(self, data: TemporalData, enc_feat: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            data: 时序数据
            enc_feat: 编码特征
            
        Returns:
            编码后的特征
        """
        # 重塑特征维度
        BN, T, _ = data.x.shape
        enc_feat = enc_feat.reshape(BN, -1, enc_feat.shape[-1])
        enc_feat = self.reshape_enc(enc_feat.transpose(1,2)).transpose(1,2)
        if self.reshape_enc_2_: enc_feat = self.reshape_enc_2(enc_feat)
        
        # 处理每个时间步的边和运动特征
        for t in range(self.historical_steps):
            data[f'edge_index_{t}'], _ = subgraph(subset=~data['padding_mask'][:, t], edge_index=data['edge_index'])
            data[f'edge_attr_{t}'] = \
                data['positions'][data[f'edge_index_{t}'][0], t] - data['positions'][data[f'edge_index_{t}'][1], t]
            data[f'motion_embed_{t}'] = enc_feat[:,t,:]
            
        if self.parallel:
            # 并行处理所有时间步
            snapshots = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                snapshots[t] = Data(x=data['x'][:, t], edge_index=edge_index, edge_attr=edge_attr,
                                    num_nodes=data['num_nodes'])
            batch = Batch.from_data_list(snapshots)
            out = self.aa_encoder(x=batch.x, t=None, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                                  bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'])
            out = out.view(self.historical_steps, out.shape[0] // self.historical_steps, -1)
        else:
            # 顺序处理每个时间步
            out = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                out[t] = self.aa_encoder(x=data['x'][:, t], t=t, edge_index=edge_index, edge_attr=edge_attr,
                                         bos_mask=data['bos_mask'][:, t], motion_embed=data[f'motion_embed_{t}'], rotate_mat=data['rotate_mat'])
            out = torch.stack(out)  # [T, N, D]
            
        # 时序编码
        out = self.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, : self.historical_steps])
        return out


class AAEncoder_hivt(MessagePassing):
    """智能体-智能体交互编码器
    
    基于MessagePassing实现节点间的消息传递和交互
    
    Args:
        historical_steps: 历史步数
        node_dim: 节点特征维度
        edge_dim: 边特征维度
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        dropout: Dropout率
        parallel: 是否并行处理
    """

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 parallel: bool = False,
                 **kwargs) -> None:
        super(AAEncoder_hivt, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel

        # 特征嵌入层
        self.center_embed = SingleInputEmbedding_hivt(in_channel=node_dim, out_channel=embed_dim)
        self.nbr_embed = MultipleInputEmbedding_hivt(in_channels=[node_dim, edge_dim], out_channel=embed_dim)

        # 多头注意力相关层
        self.lin_q = nn.Linear(embed_dim*2, embed_dim)
        self.lin_k = nn.Linear(embed_dim*2, embed_dim)
        self.lin_v = nn.Linear(embed_dim*2, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        # 门控更新相关层
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        # 归一化层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))

        # 位置编码
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                t: Optional[int],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                bos_mask: torch.Tensor,
                motion_embed: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征
            t: 时间步
            edge_index: 边索引
            edge_attr: 边属性
            bos_mask: 起始标记掩码
            motion_embed: 运动特征
            rotate_mat: 旋转矩阵
            size: 图大小
            
        Returns:
            编码后的特征
        """
        if self.parallel:
            # 并行处理
            if rotate_mat is None:
                center_embed = self.center_embed(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1))
            else:
                center_embed = self.center_embed(
                    torch.matmul(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1).unsqueeze(-2),
                                 rotate_mat.expand(self.historical_steps, *rotate_mat.shape)).squeeze(-2))
            center_embed = torch.where(bos_mask.t().unsqueeze(-1),
                                       self.bos_token.unsqueeze(-2),
                                       center_embed).view(x.shape[0], -1)
        else:
            # 顺序处理
            if rotate_mat is None:
                center_embed = self.center_embed(x)
            else:
                center_embed = self.center_embed(torch.bmm(x.unsqueeze(-2), rotate_mat).squeeze(-2))
            center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed)
            motion_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], motion_embed)

        # 残差连接和层归一化
        center_embed = center_embed + self._mha_block(self.norm1(center_embed), self.norm3(motion_embed), x, edge_index, edge_attr, rotate_mat, size)
        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        return center_embed

    def message(self,
                motion_embed_i: torch.Tensor,
                motion_embed_j: torch.Tensor,
                edge_index: Adj,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        """消息传递函数
        
        处理节点间的消息传递,计算注意力权重
        """
        if rotate_mat is None:
            nbr_embed = self.nbr_embed([x_j, edge_attr])
        else:
            if self.parallel:
                center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            else:
                center_rotate_mat = rotate_mat[edge_index[1]]
            nbr_embed = self.nbr_embed([torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                        torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)])

        # 多头注意力计算
        center_embed_i = torch.cat((center_embed_i, motion_embed_i), dim=-1)
        nbr_embed = torch.cat((nbr_embed, motion_embed_j), dim=-1)
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)

        # 注意力权重计算
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor:
        """更新函数
        
        使用门控机制更新节点特征
        """
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        return inputs + gate * (self.lin_self(center_embed) - inputs)

    def _mha_block(self,
                   center_embed: torch.Tensor,
                   motion_embed: torch.Tensor,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        # x shape: 600 X 3
        """多头注意力块"""
        center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed, motion_embed=motion_embed,
                                                    edge_attr=edge_attr, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(center_embed)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """前馈网络块"""
        return self.mlp(x)

class TemporalEncoder_hivt(nn.Module):
    """时序编码器
    
    使用Transformer架构处理时序信息
    
    Args:
        historical_steps: 历史步数
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        num_layers: 编码器层数
        dropout: Dropout率
    """

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoder_hivt, self).__init__()
        encoder_layer = TemporalEncoderLayer_hivt(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
                                                         
        # 特殊token和位置编码
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        
        # 注意力掩码
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer('attn_mask', attn_mask)
        
        # 参数初始化
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征
            padding_mask: 填充掩码
            
        Returns:
            编码后的特征
        """
        # 处理填充和特殊token
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer编码
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        """生成注意力掩码矩阵"""
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TemporalEncoderLayer_hivt(nn.Module):
    """时序编码器层
    
    包含自注意力和前馈网络
    
    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        dropout: Dropout率
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer_hivt, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                is_causal: Optional[bool] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            src: 输入特征
            src_mask: 源序列掩码
            is_causal: 是否因果
            src_key_padding_mask: 键填充掩码
            
        Returns:
            编码后的特征
        """
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """自注意力块"""
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """前馈网络块"""
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)
