"""
定义编码器/解码器层中的子层模块。包含以下几个主要组件:

1. SBI_MSA: 基于身体部位的多头自注意力层
   - 处理不同身体部位之间的关系
   - 包含位置编码和共享嵌入

2. MultiHeadAttention: 多头注意力层
   - 实现标准的多头注意力机制
   - 支持多模态输入

3. PositionwiseFeedForward: 前馈网络层
   - 用于编码器中的特征转换
   - 包含残差连接和层归一化

4. PositionwiseFeedForward_decoder: 解码器前馈网络层
   - 与编码器前馈网络结构相同
   - 专用于解码器

5. ScaledDotProductAttention: 缩放点积注意力
   - 实现注意力计算的核心机制
   - 包含温度缩放和mask处理
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SBI_MSA(nn.Module):
    """基于身体部位的多头自注意力层

    Args:
        n_head: 注意力头数
        d_model: 模型维度
        d_k: key维度
        d_v: value维度
        dropout: dropout率
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 线性变换层
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        # 输出层
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        # 正则化层
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.temperature = self.d_k**0.5

    def forward(self, src, n_person, shared_emb):
        """前向传播

        Args:
            src: 输入序列
            n_person: 人数
            shared_emb: 共享嵌入

        Returns:
            output: 注意力输出
            residual: 残差连接
            attn: 注意力权重
        """
        query, key, value = src, src, src
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = (
            query.size(0),
            query.size(1),
            key.size(1),
            value.size(1),
        )
        residual = query

        p = 5  # 身体部位数量 /  body parts
        # 线性变换并重塑维度
        q = self.w_qs(query).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(key).view(sz_b, len_k, n_head, d_k).transpose(1, 2)
        v = self.w_vs(value).view(sz_b, len_v, n_head, d_v).transpose(1, 2)

        # 处理共享嵌入
        query_emb = shared_emb.view(1, 10, n_head, d_k).transpose(1, 2)

        # 计算注意力
        attn_output_weights = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn_output_weights, dim=-1))
        output = torch.matmul(attn, v)

        # 输出处理
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        residual = output
        output = self.layer_norm(output)

        return output, residual, attn


class MultiHeadAttention(nn.Module):
    """多头注意力层

    Args:
        n_head: 注意力头数
        d_model: 模型维度
        d_k: key维度
        d_v: value维度
        d_traj_query: 轨迹查询维度
        dropout: dropout率
    """

    def __init__(self, n_head, d_model, d_k, d_v, d_traj_query=64, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 线性变换层
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        # 注意力和正则化层
        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """前向传播

        Args:
            q: 查询
            k: 键
            v: 值
            mask: 掩码

        Returns:
            q: 注意力输出
            attn: 注意力权重
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        n_modes, sz_b, len_q, len_k, len_v = (
            q.size(0),
            q.size(1),
            q.size(2),
            k.size(2),
            v.size(2),
        )
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # 线性变换和维度重塑
        q = self.w_qs(q).view(n_modes, sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(n_modes, sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(n_modes, sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # 注意力计算
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # 输出处理
        q = q.transpose(2, 3).contiguous().view(n_modes, sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """
    A two-feed-forward-layer module
    前馈网络层

    Args:
        d_in: 输入维度
        d_hid: 隐藏维度
        d_traj_query: 轨迹查询维度
        dropout: dropout率
    """

    def __init__(self, d_in, d_hid, d_traj_query=64, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播"""
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class PositionwiseFeedForward_decoder(nn.Module):
    """
    A two-feed-forward-layer module
    解码器前馈网络层

    Args:
        d_in: 输入维度
        d_hid: 隐藏维度
        d_traj_query: 轨迹查询维度
        dropout: dropout率
    """

    def __init__(self, d_in, d_hid, d_traj_query=64, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播"""
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    缩放点积注意力

    Args:
        temperature: 温度参数
        attn_dropout: 注意力dropout率
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """前向传播

        Args:
            q: 查询
            k: 键
            v: 值
            mask: 掩码

        Returns:
            output: 注意力输出
            attn: 注意力权重
        """
        attn = torch.matmul(q, k.transpose(3, 4))  # sz_b, n_head, len_q, d_k
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
