"""
PredictionModel类 - 用于轨迹和姿态预测的PyTorch Lightning模块

主要功能:
1. 初始化模型参数和评估指标
2. 实现前向传播和预测逻辑
3. 计算训练损失
4. 处理训练和验证步骤
5. 配置优化器和学习率调度器

主要组件:
- 评估指标: APE, JPE, FDE等
- 损失函数: 轨迹损失和关节损失
- 优化器: AdamW
- 学习率调度器: CosineAnnealingLR
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from metrics.t2p_metrics import APE, JPE, FDE, APE_overall, JPE_overall
from hydra.utils import instantiate
from utils_.viz import *


class PredictionModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,                  # 学习率
        warmup_epochs: int = 10,           # 预热轮数
        epochs: int = 60,                  # 总训练轮数
        weight_decay: float = 1e-4,        # 权重衰减
        output_time: int = 50,             # 输出时间步长
        dataset: str = "3dpw",         # 数据集名称
        batch_size: int = 0,               # 批次大小
        is_baseline: bool = False,         # 是否为基线模型
        num_joints: int = 15,              # 关节点数量
        viz_traj: bool = False,            # 是否可视化轨迹
        viz_joint: bool = False,           # 是否可视化关节
        viz_joint_jansang: bool = False,   # 是否可视化Jansang关节
    ) -> None:
        super(PredictionModel, self).__init__()
        
        # 保存超参数
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.output_time = output_time
        self.batch_size = batch_size
        self.is_baseline = is_baseline
        self.num_joints = num_joints
        self.viz_traj = viz_traj
        self.viz_joint = viz_joint
        self.viz_joint_jansang = viz_joint_jansang
        self.save_hyperparameters()

        # 初始化评估指标
        if self.dataset == "3dpw":
            if output_time == 20:
                metrics = MetricCollection(
                    {
                        "APE_400ms": APE(frame_idx=5),
                        "APE_800ms": APE(frame_idx=10),
                        "APE_1200ms": APE(frame_idx=15),
                        "APE_1600ms": APE(frame_idx=20),
                        "APE_overall_800ms": APE_overall(frame_idx=10),
                        "APE_overall_1600ms": APE_overall(frame_idx=20),
                        "JPE_400ms": JPE(frame_idx=5),
                        "JPE_800ms": JPE(frame_idx=10),
                        "JPE_1200ms": JPE(frame_idx=15),
                        "JPE_1600ms": JPE(frame_idx=20),
                        "JPE_overall_800ms": JPE_overall(frame_idx=10),
                        "JPE_overall_1600ms": JPE_overall(frame_idx=20),
                        "FDE_400ms": FDE(frame_idx=5),
                        "FDE_800ms": FDE(frame_idx=10),
                        "FDE_1200ms": FDE(frame_idx=15),
                        "FDE_1600ms": FDE(frame_idx=20),
                    }
                )

        # 复制评估指标到GPU
        self.val_metrics = metrics.clone()
        for k, v in self.val_metrics.items():
            self.val_metrics[k] = v.cuda()
        self.output_dir = None
        # self.historical_steps = historical_steps
        # self.future_steps = future_steps

    def forward(self, data, mode):
        """前向传播"""
        if mode == "train":
            pred_traj, gt_traj, rec, offset = self.net(data, mode)
            return pred_traj, gt_traj, rec, offset
        elif mode == "eval":
            pred_traj, gt_traj = self.net(data, mode)
            return pred_traj, gt_traj

    def predict(self, data):
        """预测函数"""
        with torch.no_grad():
            out = self.net(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        )
        return predictions, prob

    def cal_loss(self, gt_traj, pred_traj, rec, offset, data):
        """计算损失函数"""
        if not self.is_baseline:
            # 计算轨迹损失
            l2 = torch.norm(gt_traj - pred_traj, p=2, dim=-1)
            ade = l2.mean(-1)
            made_idcs = torch.argmin(ade, dim=0)

            traj_loss = l2[made_idcs, torch.arange(l2.size(1))]
            traj_loss = traj_loss[~data.padding_mask[:, -self.output_time :]].mean()

            # 计算关节损失
            joint_loss = torch.norm(
                (offset[:, 1 : self.output_time + 1] - offset[:, : self.output_time])[
                    :, :, 1:
                ]
                - rec[made_idcs, torch.arange(rec.size(1))],
                p=2,
                dim=-1,
            )
            joint_loss = joint_loss[~data.padding_mask[:, -self.output_time :]].mean()

            loss = traj_loss + joint_loss

            return {
                "loss": loss,
                "traj_loss": traj_loss.item(),
                "joint_loss": joint_loss.item(),
            }
        else:
            # 基线模型的损失计算
            N_MODES, NB, _, _, _ = rec.shape
            output_compare = (
                offset[:, 1 : self.output_time + 1, :]
                - offset[:, : self.output_time, :]
            )
            gt_rec = output_compare.unsqueeze(0).reshape(
                1, NB, self.output_time, self.num_joints, 3
            )
            l2 = torch.norm(rec - gt_rec, p=2, dim=-1)
            ade = l2.mean(-1).mean(-1)
            made_idcs = torch.argmin(ade, dim=0)
            rec = rec[made_idcs, torch.arange(rec.size(1))].reshape(
                NB, -1, self.num_joints * 3
            )
            rec_loss = torch.norm(
                rec[:, : self.output_time, :]
                - (
                    offset[:, 1 : self.output_time + 1, :]
                    - offset[:, : self.output_time, :]
                ),
                p=2,
                dim=-1,
            )
            rec_loss = rec_loss[~data.padding_mask[:, -self.output_time :]].mean()

            return {
                "loss": rec_loss,
                "rec_loss": rec_loss.item(),
            }

    def training_step(self, data, batch_idx):
        """训练步骤"""
        pred_traj, gt_traj, rec, offset = self(data, mode="train")
        losses = self.cal_loss(gt_traj, pred_traj, rec, offset, data)

        # 记录训练损失
        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.batch_size,
            )

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        """验证步骤"""
        out, gt = self(data, mode="eval")
        padding_mask = data.padding_mask[:, -self.output_time :]

        # 计算验证指标
        metrics = self.val_metrics(out, gt, padding_mask)
        
        # 可视化
        if batch_idx != 0:
            if self.viz_traj:
                viz_trajectory(out, gt, data, self.output_dir, batch_idx)
            if self.viz_joint:
                viz_joint(out, gt, data, self.output_dir, batch_idx)
            if self.viz_joint_jansang:
                if batch_idx < 10:
                    viz_joint_jansang_v2(out, gt, data, self.output_dir, batch_idx)

    def on_validation_epoch_end(self):
        """验证epoch结束时的处理"""
        final_metrics = self.val_metrics.compute()
        # print('Printing actual results!')
        # for n in final_metrics.keys():
        #     print(f'{n}: {final_metrics[n]}')
        # print('Final results done')
        for k, v in self.val_metrics.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=1,
            )
            # self.logger.log_metrics()

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 区分需要和不需要权重衰减的参数
        decay = set()
        no_decay = set()
        
        # 定义需要权重衰减的模块类型
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        
        # 定义不需要权重衰减的模块类型
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        
        # 遍历所有参数,分类到decay或no_decay
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        
        # 获取所有参数
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        
        # 验证参数分类的正确性
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        # 创建优化器参数组
        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # 初始化优化器
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # 初始化学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=int(self.epochs * 1.1), eta_min=0.0
        )

        return [optimizer], [scheduler]
