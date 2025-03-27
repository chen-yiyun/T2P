"""
定义模型的主要层结构。包含以下组件:

1. DecoderLayer: 解码器层
   - 包含多头注意力机制和前馈网络
   - 用于生成预测序列

2. TBIFormerBlock: Transformer编码器块
   - 包含基于身体部位的多头自注意力(SBI-MSA)
   - 包含前馈网络层
   - 用于提取时空特征
"""

import torch.nn as nn
from .sublayers import (
    PositionwiseFeedForward,
    SBI_MSA, 
    MultiHeadAttention,
    PositionwiseFeedForward_decoder,
)


class DecoderLayer(nn.Module):
    """
    Compose with three layers
    解码器层,由多头注意力和前馈网络组成

    Args:
        d_model: 模型维度
        d_inner: 内层维度
        n_head: 注意力头数
        d_k: key维度
        d_v: value维度
        d_traj_query: 轨迹查询维度
        dropout: dropout率
    """

    def __init__(
        self, d_model, d_inner, n_head, d_k, d_v, d_traj_query=64, dropout=0.1
    ):
        super(DecoderLayer, self).__init__()
        # 多头注意力层
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, d_traj_query=d_traj_query, dropout=dropout
        )
        # 前馈网络层
        self.pos_ffn = PositionwiseFeedForward_decoder(
            d_model, d_inner, d_traj_query=d_traj_query, dropout=dropout
        )

    def forward(self, dec_input, enc_output):
        """
        前向传播
        
        Args:
            dec_input: 解码器输入
            enc_output: 编码器输出
            
        Returns:
            dec_output: 解码器输出
            dec_enc_attn: 注意力权重
        """
        dec_output, dec_enc_attn = self.enc_attn(dec_input, enc_output, enc_output)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_enc_attn


class TBIFormerBlock(nn.Module):
    """
    Transformer编码器块
    
    Args:
        d_model: 模型维度
        d_inner: 内层维度
        n_head: 注意力头数
        d_k: key维度
        d_v: value维度
        dropout: dropout率
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TBIFormerBlock, self).__init__()
        # 基于身体部位的多头自注意力层
        self.sbi_msa = SBI_MSA(n_head, d_model, d_k, d_v, dropout=dropout)
        # 前馈网络层
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, n_person, emb):
        """
        前向传播
        
        Args:
            enc_input: 编码器输入
            n_person: 人数
            emb: 嵌入向量
            
        Returns:
            enc_output: 编码器输出
            enc_slf_attn: 自注意力权重
        """
        enc_output, residual, enc_slf_attn = self.sbi_msa(enc_input, n_person, emb)
        enc_output = self.pos_ffn(enc_output)
        enc_output += residual
        return enc_output, enc_slf_attn
