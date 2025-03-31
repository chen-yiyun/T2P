"""
T2P模型的评估指标模块

包含以下指标类:
- APE: 绝对位置误差(单帧)
- APE_overall: 绝对位置误差(整体)
- JPE: 关节位置误差(单帧)
- JPE_overall: 关节位置误差(整体)
- FDE: 最终位移误差
- ADE_TR: 平均轨迹误差
- FDE_TR: 最终轨迹误差
"""

from typing import Any, Callable, Dict, Optional
import torch
import numpy as np
from torchmetrics import Metric


class APE(Metric):
    """
    绝对位置误差(单帧)
    计算预测姿态与目标姿态在指定帧的位置差异
    ---
    对齐的每关节平均位置误差用于评估预测的局部运动。
    L2 髋关节坐标系中每个关节的距离在给定时间步长内对所有关节进行平均。
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,  # 指定计算误差的帧索引
        method="mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1000  # 误差放大系数
        super(APE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.method = method
        # 添加状态变量
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, target, padding_mask):
        """更新误差统计
        Args:
            outputs: 预测姿态 [B*N, T, 15, 3]
            target: 目标姿态 [B*N, T, 15, 3]
            padding_mask: 填充掩码
        """
        # 减去根节点坐标,转为相对位置
        if self.method == "hip":
            outputs = outputs - outputs[:, :, 0:1, :]
            target = target - target[:, :, 0:1, :]
        elif self.method == "mean":
            outputs = outputs - outputs.mean(dim=2, keepdim=True)
            target = target - target.mean(dim=2, keepdim=True)
        elif self.method == "weighted":
            # 加权平均法：为不同关节分配权重
            # 定义权重（示例权重，可以根据需要调整）
            # 假设有13个关节点
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
            # 保持维度匹配
            outputs = outputs - torch.tensordot(
                outputs, weights, dims=([2], [0])
            ).unsqueeze(-2)
            target = target - torch.tensordot(
                target, weights, dims=([2], [0])
            ).unsqueeze(-2)
        else:
            raise ValueError(f"Invalid method: {self.method}")
        # 创建帧掩码
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, self.frame_idx - 1] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())

        # 计算L2误差
        err = torch.norm(target[padding_mask] - outputs[padding_mask], p=2, dim=-1)
        err_frames = err.mean(-1)
        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]

    def compute(self) -> torch.Tensor:
        """计算平均误差"""
        return (self.sum / self.count) * self.scale


class APE_overall(Metric):
    """
    绝对位置误差(整体)
    计算预测姿态与目标姿态在所有帧的平均位置差异
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,
        method="mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1000
        self.method = method
        super(APE_overall, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, target, padding_mask):
        """更新误差统计"""
        if self.method == "hip":
            outputs = outputs - outputs[:, :, 0:1, :]
            target = target - target[:, :, 0:1, :]
        elif self.method == "mean":
            outputs = outputs - outputs.mean(dim=2, keepdim=True)
            target = target - target.mean(dim=2, keepdim=True)
        elif self.method == "weighted":
            # 加权平均法：为不同关节分配权重
            # 定义权重（示例权重，可以根据需要调整）
            # 假设有13个关节点
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
            outputs = outputs - torch.tensordot(
                outputs, weights, dims=([2], [0])
            ).unsqueeze(-2)
            target = target - torch.tensordot(
                target, weights, dims=([2], [0])
            ).unsqueeze(-2)
        else:
            raise ValueError(f"Invalid method: {self.method}")

        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, : self.frame_idx] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())

        err = torch.norm(target[padding_mask] - outputs[padding_mask], p=2, dim=-1)
        err_frames = err.mean(-1)
        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count) * self.scale


class JPE(Metric):
    """
    关节位置误差(单帧)
    计算预测姿态与目标姿态在指定帧的关节位置差异
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,
        method="mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1000
        self.method = method
        super(JPE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, target, padding_mask):
        """更新误差统计"""
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, self.frame_idx - 1] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())

        err = torch.norm(target[padding_mask] - outputs[padding_mask], p=2, dim=-1)
        err_frames = err.mean(-1)

        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count) * self.scale


class JPE_overall(Metric):
    """
    关节位置误差(整体)
    计算预测姿态与目标姿态在所有帧的平均关节位置差异
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,
        method="mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1000
        self.method = method
        super(JPE_overall, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, target, padding_mask):
        """更新误差统计"""
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, : self.frame_idx] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())

        err = torch.norm(target[padding_mask] - outputs[padding_mask], p=2, dim=-1)
        err_frames = err.mean(-1)
        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count) * self.scale


class FDE(Metric):
    """
    最终位移误差
    计算预测姿态与目标姿态在最后一帧的位置差异
    ---
    最终距离误差通过计算给定时间步长的髋关节L2 距离来评估预测的全局轨迹。
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,
        method="mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1000
        self.method = method
        super(FDE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, target, padding_mask):
        """更新误差统计"""
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, self.frame_idx - 1] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())
        target, outputs = target[padding_mask], outputs[padding_mask]
        if self.method == "hip":
            err_frames = torch.norm(target[:, 0] - outputs[:, 0], p=2, dim=-1)
        elif self.method == "mean":
            # 计算所有关节的中心点
            err_frames = torch.norm(
                target.mean(dim=1) - outputs.mean(dim=1), p=2, dim=-1
            )  # 计算L2距离
        elif self.method == "weighted":
            # 加权平均法：为不同关节分配权重
            # 定义权重（示例权重，可以根据需要调整）
            # 假设有13个关节点
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
            err_frames = torch.norm(
                torch.tensordot(target, weights, dims=([1], [0]))
                - torch.tensordot(outputs, weights, dims=([1], [0])),
                p=2,
                dim=-1,
            )
        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count) * self.scale


class ADE_TR(Metric):
    """
    平均轨迹误差
    计算预测轨迹与目标轨迹在所有帧的平均位置差异
    ---
    未用到
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,
        method="mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1
        self.method = method
        super(ADE_TR, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, target, padding_mask):
        """更新误差统计"""
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, : self.frame_idx] = 1
        if self.frame_idx == -1:
            frame_mask[:, self.frame_idx] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())
        target = target[padding_mask]
        outputs = outputs[padding_mask]
        err = torch.norm(target[:, 0, :2] - outputs[:, 0, :2], p=2, dim=-1)
        self.sum += err.sum()
        self.count += err.shape[0]

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count) * self.scale


class FDE_TR(Metric):
    """
    最终轨迹误差
    计算预测轨迹与目标轨迹在最后一帧的位置差异
    ---
    未用到
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,
        method="mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1
        self.method = method
        super(FDE_TR, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, target, padding_mask):
        """更新误差统计"""
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        if self.frame_idx != -1:
            frame_mask[:, self.frame_idx - 1] = 1
        else:
            frame_mask[:, -1] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())
        target, outputs = target[padding_mask], outputs[padding_mask]
        if self.method == "hip":
            err_frames = torch.norm(target[:, 0, :2] - outputs[:, 0, :2], p=2, dim=-1)
        elif self.method == "mean":
            err_frames = torch.norm(
                target.mean(dim=1)[:, 2] - outputs.mean(dim=1)[:, 2], p=2, dim=-1
            )
        elif self.method == "weighted":
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
            err_frames = torch.norm(
                torch.tensordot(target, weights, dims=([1], [0]))
                - torch.tensordot(outputs, weights, dims=([1], [0])),
                p=2,
                dim=-1,
            )
        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count) * self.scale
