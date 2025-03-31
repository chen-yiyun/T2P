"""
定义Transformer模型
这是一个用于人体动作预测的模型,包含以下主要组件:

1. 时间和身份编码器(Tem_ID_Encoder):
- 为序列添加时间位置编码和身份编码
- 帮助模型区分不同时间步和不同人物

2. TBIFormer编码器(TBIFormerEncoder):
- 基于Transformer的编码器
- 处理时空特征

3. 解码器(Decoder):
- 基于Transformer的解码器
- 生成预测序列

4. 主模型T2P:
- 输入:人体姿态序列
- 输出:预测的未来姿态
- 包含轨迹预测和局部姿态预测两个分支
"""

import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .layers import DecoderLayer, TBIFormerBlock
from models import GlobalInteractor_hivt
from models import LocalEncoder_hivt
from models import MLPDecoder_hivt
from utils import TemporalData

import itertools
import torch
import numpy as np
import torch_dct as dct  # 用于离散余弦变换


def temporal_partition(src, opt):
    """时间分割函数,将输入序列按时间步长分割"""
    src = src[:, :, 1:]
    B, N, L, _ = src.size()
    stride = 1
    fn = int((L - opt.kernel_size) / stride + 1)
    idx = (
        np.expand_dims(np.arange(opt.kernel_size), axis=0)
        + np.expand_dims(np.arange(fn), axis=1) * stride
    )
    return idx


class Tem_ID_Encoder(nn.Module):
    """
    时间和身份编码器
    为序列添加时间位置编码和身份编码
    """

    def __init__(self, d_model, dropout=0.1, max_t_len=200, max_a_len=20):
        super(Tem_ID_Encoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = self.build_pos_enc(max_t_len)  # 构建位置编码
        self.register_buffer("pe", pe)
        ie = self.build_id_enc(max_a_len)  # 构建身份编码
        self.register_buffer("ie", ie)

    def build_pos_enc(self, max_len):
        """构建正弦位置编码"""
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def build_id_enc(self, max_len):
        """构建身份编码"""
        ie = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model)
        )
        ie[:, 0::2] = torch.sin(position * div_term)
        ie[:, 1::2] = torch.cos(position * div_term)
        ie = ie.unsqueeze(0)
        return ie

    def get_pos_enc(self, num_a, num_p, num_t, t_offset):
        """获取位置编码"""
        pe = self.pe[:, t_offset : num_t + t_offset]
        pe = pe.repeat(1, num_a * num_p, 1)
        return pe

    def get_id_enc(self, num_p, num_t, i_offset, id_enc_shuffle):
        """获取身份编码"""
        ie = self.ie[:, id_enc_shuffle]
        ie = ie.repeat_interleave(num_p * num_t, dim=1)
        return ie

    def forward(self, x, num_a, num_p, num_t, t_offset=0, i_offset=0):
        """
        前向传播
        num_a: 人数 number of person,
        num_p: 身体部位数 number of body parts
        num_t: 时间长度 length of time
        t_offset: 时间偏移 time offset
        i_offset: 身份偏移 identity offset
        """
        index = list(np.arange(0, num_p))
        id_enc_shuffle = random.choices(index, k=num_a)
        pos_enc = self.get_pos_enc(num_a, num_p, num_t, t_offset)
        id_enc = self.get_id_enc(num_p, num_t, i_offset, id_enc_shuffle)
        x = x + pos_enc + id_enc  # 添加时间编码和身份编码

        return self.dropout(x)


class TBIFormerEncoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    TBIFormer编码器
    基于Transformer的编码器,处理时空特征
    """

    def __init__(
        self,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        dropout=0.1,
        n_position=1000,
        device="cuda",
        kernel_size=10,
    ):
        super().__init__()
        self.embeddings = Tem_ID_Encoder(
            d_model, dropout=dropout, max_t_len=n_position, max_a_len=20
        )  # 时间和身份编码器
        self.layer_stack = nn.ModuleList(
            [
                TBIFormerBlock(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        self.embeddings_table = nn.Embedding(10, d_k * n_head)

    def forward(self, src, n_person, return_attns=False):
        """
        前向传播
        src: 输入序列 [B,N,T,D]
        """
        enc_attn_list = []
        sz_b, n, p, t, d = src.size()

        src = src.reshape(sz_b, -1, d)

        enc_in = self.embeddings(src, n, p, t)  # 时间和身份编码

        enc_output = enc_in
        for enc_layer in self.layer_stack:
            enc_output, enc_attn = enc_layer(
                enc_output, n_person, self.embeddings_table.weight
            )
            enc_attn_list += [enc_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_attn_list

        return enc_output


class Decoder(nn.Module):
    """
    解码器
    基于Transformer的解码器,生成预测序列
    """

    def __init__(
        self,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        d_traj_query=64,
        dropout=0.1,
        device="cuda",
    ):
        super().__init__()
        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    d_traj_query=d_traj_query,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, trg_seq, enc_output, return_attns=False):
        """
        前向传播
        trg_seq: 目标序列
        enc_output: 编码器输出
        """
        dec_enc_attn_list = []
        dec_output = trg_seq  # bs * person, 3 * person + input_frames, dim=128
        layer = 0
        for dec_layer in self.layer_stack:
            layer += 1
            dec_output, dec_enc_attn = dec_layer(dec_output, enc_output)
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_enc_attn_list
        return dec_output


def body_partition(mydata, index):
    """
    身体部位分割函数
    将人体关节点分组为不同的身体部位
    """
    bn, seq_len, _ = mydata.shape
    mydata = mydata.reshape(bn, seq_len, -1, 3)  # 96, 50, 15, 3
    out = torch.zeros(bn, seq_len, len(index), 3).to(mydata.device)  # x, 12, 3, 35
    for i in range(len(index)):
        temp1 = mydata[:, :, index[i], :].reshape(-1, len(index[i]), 3).transpose(1, 2)
        temp2 = F.avg_pool1d(temp1, kernel_size=5, padding=1)
        temp2 = temp2.transpose(1, 2).reshape(bn, seq_len, -1, 3)
        out[:, :, i, :] = temp2[:, :, 0, :]
    return out


class T2P(nn.Module):
    """
    主模型T2P(Trajectory to Pose)
    包含轨迹预测和局部姿态预测两个分支
    """

    def __init__(
        self,
        input_dim=128,
        d_model=512,
        d_inner=1024,
        n_layers=3,
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.2,
        device="cuda",
        kernel_size=10,
        d_traj_query=64,
        opt=None,
    ):

        super().__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.kernel_size = opt.kernel_size
        self.device = device
        self.d_model = d_model
        self.output_time = opt.output_time
        self.input_time = opt.input_time
        self.num_joints = opt.num_joints
        self.dataset = opt.dataset
        self.method = opt.method
        self.sampling_method = opt.sampling_method

        # 2D卷积层,用于特征提取
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=input_dim,
                kernel_size=(1, opt.kernel_size),
                stride=(1, 1),
                bias=False,
            ),
            nn.ReLU(inplace=False),
        )

        # 编码器
        self.encoder = TBIFormerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            device=self.device,
            kernel_size=kernel_size,
        )

        # 解码器
        self.decoder = Decoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            d_traj_query=d_traj_query,
            dropout=dropout,
            device=self.device,
        )

        # MLP层
        kernel_size1 = int(kernel_size / 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size2 = int(kernel_size / 2)
        else:
            kernel_size2 = int(kernel_size / 2 + 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_joints * 3,
                out_channels=d_model,
                kernel_size=kernel_size1,
                bias=False,
            ),
            nn.ReLU(inplace=False),
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size2,
                bias=False,
            ),
            nn.ReLU(inplace=False),
        )

        # 投影层
        self.proj_inverse = nn.Linear(d_model, (self.num_joints - 1) * 3)
        self.l1 = nn.Linear(d_model, (d_model // 4) * self.output_time)
        self.l2 = nn.Linear(d_model // 4, d_model)
        self.query_linear = nn.Linear(d_model + d_traj_query, d_model)

        # HiVT组件(用于轨迹预测)
        historical_steps = opt.input_time
        self.future_steps = opt.output_time
        node_dim, edge_dim = 3, 3  # number of dimensions
        (
            num_heads,
            hivt_dropout,
            num_temporal_layers,
            local_radius,
            parallel,
            num_modes,
        ) = (8, 0.1, 4, 50, False, opt.num_modes)
        num_global_layers, rotate = 3, True
        embed_dim = opt.hivt_embed_dim
        reshape_dim = (self.input_time - self.kernel_size) * 5

        # HiVT的三个主要组件
        self.local_encoder_traj = LocalEncoder_hivt(
            reshape_dim=reshape_dim,
            historical_steps=historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=hivt_dropout,
            num_temporal_layers=num_temporal_layers,
            local_radius=local_radius,
            parallel=parallel,
            enc_feat_dim=input_dim,
        )
        self.global_interactor_traj = GlobalInteractor_hivt(
            historical_steps=historical_steps,
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            num_modes=num_modes,
            num_heads=num_heads,
            num_layers=num_global_layers,
            dropout=hivt_dropout,
            rotate=rotate,
        )
        self.decoder_traj = MLPDecoder_hivt(
            local_channels=embed_dim,
            global_channels=embed_dim,
            future_steps=self.future_steps,
            num_modes=num_modes,
            uncertain=False,
        )

        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert (
            d_model == input_dim
        ), "To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same."  # "维度必须匹配以便进行残差连接"

    def forward(self, batch_data, mode):
        """
        前向传播
        batch_data: 输入数据批次
        mode: 'train'或'eval'模式

        return:
            predicted_motion: 预测的轨迹
            gt: 真实轨迹
            rec: 局部姿态预测
            offset_output: 相对位置预测
        """
        # 准备输入数据
        input_seq, output_seq = (
            batch_data.input_seq.clone(),
            batch_data.output_seq.clone(),
        )
        if len(input_seq.shape) == 3:
            bn, t, d = input_seq.shape
            input_seq = input_seq.reshape(bn // 25, 25, t, d)
        if len(output_seq.shape) == 3:
            bn, t, d = output_seq.shape
            output_seq = output_seq.reshape(bn // 25, 25, t, d)
        B, N, _, D = input_seq.shape
        hip_joint_idx = 0  # 髋关节索引 / Out of 15 joints, hip joint is idx 0

        # 处理输入序列
        input_ = input_seq.view(-1, self.input_time, input_seq.shape[-1])
        output_ = output_seq.view(
            output_seq.shape[0] * output_seq.shape[1], -1, input_seq.shape[-1]
        )

        # 计算相对位置
        input_ = input_.reshape(-1, self.opt.input_time, self.num_joints, 3)
        if self.method == "hip":
            # 髋关节位置作为参考点
            input_hip = (
                input_[:, :, 0, :].unsqueeze(-2).repeat(1, 1, self.num_joints, 1)
            )
            offset = input_ - input_hip
        elif self.method == "mean":
            # 计算均值并广播到所有关节
            input_mean = (
                input_.mean(dim=2).unsqueeze(-2).repeat(1, 1, self.num_joints, 1)
            )
            offset = input_ - input_mean  # 计算相对位置
        elif self.method == "weighted":
            # 加权平均法：为不同关节分配权重
            # 定义权重（示例权重，可以根据需要调整）
            # 假设有13个关节点
            # 定义权重方案（索引: 权重）
            weight_scheme = {
                0: 0.15,
                1: 0.15,
                2: 0.08,
                3: 0.08,
                4: 0.05,
                5: 0.05,
                6: 0.15,
                7: 0.10,
                8: 0.10,
                9: 0.04,
                10: 0.04,
                11: 0.01,
                12: 0.01,
            }
            weights = torch.tensor(
                [weight_scheme[i] for i in range(len(weight_scheme))]
            ).cuda()
            weights = weights / weights.sum()  # 归一化权重
            # print(f"Current device: {torch.cuda.current_device()}")
            # print(f"input device: {input_.device}")
            # print(f"Weights device: {weights.device}")
            # 确保 weights 在与 input_ 相同的设备上
            # input_= input_.to("cuda")
            # output_ = output_.to("cuda")
            # weights = weights.to("cuda")  # 将 weights 移动到 input_ 的设备上
            # 计算加权平均位置并广播到所有关节
            input_weighted = (
                torch.tensordot(input_, weights, dims=([2], [0]))
                .unsqueeze(-2)
                .repeat(1, 1, self.num_joints, 1)
            )
            offset = input_ - input_weighted
        else:
            raise ValueError(f"Invalid method: {self.method}")

        # 计算时间位移
        offset = offset.reshape(-1, self.opt.input_time, input_seq.shape[-1])
        offset = (
            offset[:, 1 : self.opt.input_time, :]
            - offset[:, : self.opt.input_time - 1, :]
        )
        src = dct.dct(offset)  # DCT变换
        enc_feat = self.forward_encode_body(src, N)

        #################### 轨迹预测 ################### 轨迹预测
        pred_trajectory, pi, traj_feats = self.traj_forward(batch_data, enc_feat)
        num_modes = traj_feats.shape[0]
        inverse_rotMat = torch.linalg.inv(batch_data.rotate_mat)
        inverse_rotMat = inverse_rotMat.unsqueeze(0).repeat(num_modes, 1, 1, 1)
        predicted_motion = torch.matmul(pred_trajectory, inverse_rotMat)
        predicted_motion = (
            (
                predicted_motion
                + batch_data.positions[:, self.opt.input_time - 1]
                .unsqueeze(0)
                .unsqueeze(-2)
            )
            .unsqueeze(-2)
            .repeat(1, 1, 1, self.num_joints, 1)
        )
        ###############################################################

        traj_feats = traj_feats.clone().detach()

        #################### 局部姿态预测 ####################
        rec_ = self.forward_local(src, N, traj_feats, enc_feat)
        rec = dct.idct(rec_)  # 逆DCT变换
        rec = rec.reshape(
            num_modes, N * B, self.opt.output_time, self.num_joints - 1, 3
        )

        # 处理真实轨迹
        gt_trajectory = batch_data.y.unsqueeze(0)
        output_ = output_[:, :, :].reshape(
            -1, self.opt.output_time + 1, self.num_joints, 3
        )  #   Relative position to hip joint
        if self.method == "hip":
            # 髋关节位置作为参考点
            output_hip = (
                output_[:, :, 0, :].unsqueeze(-2).repeat(1, 1, self.num_joints, 1)
            )
            offset_output = output_ - output_hip
        elif self.method == "mean":
            # 计算均值并广播到所有关节
            output_mean = (
                output_.mean(dim=2).unsqueeze(-2).repeat(1, 1, self.num_joints, 1)
            )
            offset_output = output_ - output_mean
        elif self.method == "weighted":
            # 加权平均法：为不同关节分配权重
            # 定义权重（示例权重，可以根据需要调整）
            # 假设有13个关节点
            # 定义权重方案（索引: 权重）
            weight_scheme = {
                0: 0.15,
                1: 0.15,
                2: 0.08,
                3: 0.08,
                4: 0.05,
                5: 0.05,
                6: 0.15,
                7: 0.10,
                8: 0.10,
                9: 0.04,
                10: 0.04,
                11: 0.01,
                12: 0.01,
            }
            weights = torch.tensor(
                [weight_scheme[i] for i in range(len(weight_scheme))]
            ).cuda()
            weights = weights / weights.sum()  # 归一化权重
            # 确保 weights 在与 output_ 相同的设备上
            # weights = weights.to(output_.device)  # 将 weights 移动到 output_ 的设备上
            # 计算加权平均位置并广播到所有关节
            output_weighted = (
                torch.tensordot(output_, weights, dims=([2], [0]))
                .unsqueeze(-2)
                .repeat(1, 1, self.num_joints, 1)
            )
            offset_output = output_ - output_weighted
        else:
            raise ValueError(f"Invalid method: {self.method}")

        # 评估模式
        if mode == "eval":
            if self.sampling_method == "ade":
                # 使用平均位移误差(ADE)选择最佳预测
                l2 = torch.norm(gt_trajectory - pred_trajectory, p=2, dim=-1)
                mask_ = ~batch_data.padding_mask[:, -self.output_time :]
                masked_l2 = l2 * mask_.unsqueeze(0)
                sum_masked_l2 = masked_l2.sum(dim=2)
                sum_mask = mask_.sum(dim=1)
                sum_mask[sum_mask == 0] += 1
                average_masked_l2 = sum_masked_l2 / sum_mask.unsqueeze(0)
                made_idcs = torch.argmin(average_masked_l2, dim=0)
            elif self.sampling_method == "fde":
                # 使用最终位移误差(FDE)选择最佳预测
                l2 = torch.norm(gt_trajectory - pred_trajectory, p=2, dim=-1)
                mask_ = ~batch_data.padding_mask[:, -self.output_time :]

                for agentIdx in range(mask_.shape[0]):
                    if sum(mask_[agentIdx]) == 0:
                        continue
                    max_idx = torch.where(mask_[agentIdx] == True)[0].max()
                    mask_[agentIdx, :] = False
                    mask_[agentIdx, max_idx] = True

                masked_l2 = l2 * mask_.unsqueeze(0)
                sum_masked_l2 = masked_l2.sum(dim=2)
                sum_mask = mask_.sum(dim=1)
                sum_mask[sum_mask == 0] += 1
                average_masked_l2 = sum_masked_l2 / sum_mask.unsqueeze(0)
                made_idcs = torch.argmin(average_masked_l2, dim=0)

            # 生成最终预测
            offset_output = offset_output.clone().detach()
            offset_output = offset_output.unsqueeze(0).repeat(num_modes, 1, 1, 1, 1)
            results = offset_output[:, :, :1, 1:]
            for i in range(1, self.opt.output_time + 1):
                results = torch.cat(
                    [
                        results,
                        offset_output[:, :, :1, 1:]
                        + torch.sum(rec[:, :, :i, :], dim=2, keepdim=True),
                    ],
                    dim=2,
                )

            predicted_motion[:, :, :, 1:, :] = (
                predicted_motion[:, :, :, 1:, :] + results[:, :, 1:]
            )
            predicted_motion = predicted_motion.reshape(
                num_modes, B * N, self.opt.output_time, self.num_joints, 3
            )
            predicted_motion = predicted_motion[made_idcs, torch.arange(B * N)]
            gt = output_.view(B, N, -1, self.num_joints, 3)[:, :, 1:, ...]

            return predicted_motion, gt.reshape(
                B * N, self.opt.output_time, self.num_joints, 3
            )
        elif mode == "train":
            return pred_trajectory, gt_trajectory, rec, offset_output

    def forward_encode_body(self, src, n_person):
        """
        src_seq:  B*N, T, J*3
        身体编码前向传播
        src: 输入序列
        n_person: 人数

        return:
            enc_out: 编码器输出
        """
        bn = src.shape[0]  # 获取输入序列的批次大小
        bs = int(bn / n_person)  # 计算每个个体的批次大小

        # 身体部位分割
        if self.dataset == "3dpw":
            index = [
                [0, 2, 4],
                [1, 3, 5],
                [8, 10, 12],
                [7, 9, 11],
                [6, 7, 8],
            ]  # 5个身体部位

        # 处理身体部位序列 / multi-person body parts sequence
        part_seq = body_partition(src, index).permute(0, 3, 2, 1)
        mpbp_seq = (
            self.conv2d(part_seq).permute(0, 2, 3, 1).reshape(bs, n_person, 5, -1, 128)
        )

        # TBIFormer编码
        enc_out = self.encoder(mpbp_seq, n_person)
        return enc_out

    def forward_local(self, src, n_person, traj_query, enc_out):
        """
        src_seq:  B*N, T, J*3
        局部姿态预测前向传播

        return:
            dec_out: 解码器输出
        """
        num_modes = traj_query.shape[0]
        bn = src.shape[0]
        bs = int(bn / n_person)

        # ======= Transformer Decoder ============
        # 准备查询
        src_query = src.transpose(1, 2)[
            :, :, -self.kernel_size :
        ].clone()  # the last sub-sequence for query
        global_body_query = self.mlp(src_query).reshape(bs, n_person, -1)
        global_body_query = global_body_query.unsqueeze(0).repeat(num_modes, 1, 1, 1)
        enc_out = enc_out.unsqueeze(0).repeat(num_modes, 1, 1, 1)
        traj_query = traj_query.reshape(num_modes, bs, n_person, -1)
        new_query = torch.cat((global_body_query, traj_query), dim=-1)
        new_query = self.query_linear(new_query)

        # 解码
        dec_output = self.decoder(new_query, enc_out, False)
        dec_output = dec_output.reshape(num_modes, bn, 1, -1)

        # 全连接层处理
        dec_output = self.l1(dec_output)
        dec_output = dec_output.view(num_modes, bn, self.future_steps, -1)
        dec_output = self.l2(dec_output)
        dec_out = self.proj_inverse(dec_output)
        return dec_out

    def traj_forward(self, input_traj_temporalData, enc_feat):
        """
        轨迹预测前向传播
        使用HiVT组件进行预测

        return:
            y_hat: 预测的轨迹
            pi: 预测的轨迹索引
            out_feature: 预测的轨迹特征
        """
        local_embed = self.local_encoder_traj(
            data=input_traj_temporalData, enc_feat=enc_feat
        )
        global_embed = self.global_interactor_traj(
            data=input_traj_temporalData, local_embed=local_embed
        )
        y_hat, pi, out_feature = self.decoder_traj(
            local_embed=local_embed, global_embed=global_embed
        )
        return y_hat, pi, out_feature
