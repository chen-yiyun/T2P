"""
可视化工具
"""

# 导入必要的库
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import glob
from PIL import Image
import os

# 定义人体骨骼颜色和连接关系
HUMAN_COLORS = [
    "orangered",
    "limegreen",
    "deepskyblue",
    "cyan",
    "skyblue",
    "navy",
    "magenta",
    "darkturquoise",
    "olive",
    "dimgray",
    "darkorange",
    "lightcoral",
    "lime",
    "yellowgreen",
    "peru",
    "chocolate",
    "orangered",
    "navy",
    "mediumturquoise",
    "crimson",
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "skyblue",
    "olive",
    "dimgray",
    "darkorange",
]

# 定义15关节点骨骼连接关系
TBI15_BONES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [7, 9],
        [9, 10],
        [10, 11],
        [7, 12],
        [12, 13],
        [13, 14],
    ]
)

# 定义3DPW数据集骨骼连接关系
EDGES_3DPW = np.array(
    [
        (0, 1),
        (1, 8),
        (8, 7),
        (7, 0),
        (0, 2),
        (2, 4),
        (1, 3),
        (3, 5),
        (7, 9),
        (9, 11),
        (8, 10),
        (10, 12),
        (6, 7),
        (6, 8),
    ]
)


def viz_trajectory(pred, gt, data, output_dir, batch_idx):
    """
    可视化轨迹预测结果
    Args:
        pred: 预测轨迹 [B*N, T, 15, 3]
        gt: 真实轨迹 [B*N, T, 15, 3]
        data: 数据对象
        output_dir: 输出目录
        batch_idx: 批次索引
    """
    # 创建输出目录
    print(f"Visualizing trajectory of batch {batch_idx}")
    if not os.path.isdir(os.path.join(output_dir, "viz_results", "trajectory")):
        os.makedirs(os.path.join(output_dir, "viz_results", "trajectory"))
    # pred: B*N, T, 15, 3
    # 获取数据维度
    B, N, T, XYZ_3 = data.output_seq.shape

    # 处理掩码和数据
    padding_mask_past = (
        ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:, :, : -(T - 1)]
        .cpu()
        .numpy()
    )
    padding_mask_fut = (
        ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:, :, -(T - 1) :]
        .cpu()
        .numpy()
    )
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3 // 3, 3)[:, :, :, 0].cpu().numpy()
    y_gt = gt[:, :, 0].reshape(B, N, T - 1, 3).cpu().numpy()
    y_pred = pred[:, :, 0].reshape(B, N, T - 1, 3).cpu().detach().numpy()

    # 遍历每个场景
    for scene_idx in range(B):
        if padding_mask_fut[scene_idx].sum() <= 1:
            continue
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        xy_mean = np.zeros((0, 2))

        # 遍历每个agent
        for agent_idx in range(N):
            if padding_mask_fut[scene_idx, agent_idx].sum() <= 1:
                continue
            x_gt_masked = x_gt[scene_idx, agent_idx][
                padding_mask_past[scene_idx, agent_idx]
            ]
            y_gt_masked = y_gt[scene_idx, agent_idx][
                padding_mask_fut[scene_idx, agent_idx]
            ]
            y_pred_masked = y_pred[scene_idx, agent_idx][
                padding_mask_fut[scene_idx, agent_idx]
            ]

            # 绘制轨迹
            ax.plot(
                x_gt_masked[:, 0],
                x_gt_masked[:, 1],
                "ko-",
                linewidth=0.25,
                markersize=0.5,
            )
            ax.plot(
                y_gt_masked[:, 0],
                y_gt_masked[:, 1],
                "bo-",
                linewidth=0.25,
                markersize=0.5,
            )
            ax.plot(
                y_pred_masked[:, 0],
                y_pred_masked[:, 1],
                "ro-",
                linewidth=0.25,
                markersize=0.5,
            )
            xy_mean = np.concatenate(
                (xy_mean, np.expand_dims(x_gt_masked.mean(0)[:2], axis=0)), axis=0
            )

        # 设置坐标轴范围和标签
        ax.set_xlim([xy_mean.mean(0)[0] - 5, xy_mean.mean(0)[0] + 5])
        ax.set_ylim([xy_mean.mean(0)[1] - 5, xy_mean.mean(0)[1] + 5])
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # 保存图像
        fig.savefig(
            f"{output_dir}/viz_results/trajectory/batch_{batch_idx}_scene_{scene_idx}.png"
        )
        plt.close()
        plt.cla()


def viz_joint(pred, gt, data, output_dir, batch_idx, bones=None):
    """
    可视化关节点预测结果
    Args:
        pred: 预测关节点 [B*N, T, 15, 3]
        gt: 真实关节点 [B*N, T, 15, 3]
        data: 数据对象
        output_dir: 输出目录
        batch_idx: 批次索引
        bones: 骨骼连接关系
    """
    # print(f"Visualizing trajectory of batch {batch_idx}")
    # 创建输出目录
    if not os.path.isdir(os.path.join(output_dir, "viz_results", "joint")):
        os.makedirs(os.path.join(output_dir, "viz_results", "joint"))
    # pred: B*N, T, 15, 3
    # 获取数据维度和处理数据
    B, N, T, XYZ_3 = data.output_seq.shape
    padding_mask_past = (
        ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:, :, : -(T - 1)]
        .cpu()
        .numpy()
    )
    padding_mask_fut = (
        ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:, :, -(T - 1) :]
        .cpu()
        .numpy()
    )
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3 // 3, 3)[:, :, :, 0].cpu().numpy()
    y_gt = gt.reshape(B, N, T - 1, XYZ_3 // 3, 3).cpu().numpy()
    y_pred = pred.reshape(B, N, T - 1, XYZ_3 // 3, 3).cpu().detach().numpy()

    # 转换数据类型
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    B, N, T, XYZ_3 = data.output_seq.shape
    # 获取总帧数
    TOTAL_T = data.input_seq.shape[2] + data.output_seq.shape[2] - 1
    xy_sizes = [4.5]
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3 // 3, 3).detach().cpu().numpy()
    y_gt = (
        data.output_seq.reshape(B, N, -1, XYZ_3 // 3, 3)
        .detach()
        .cpu()
        .numpy()[:, :, 1:]
    )
    padding_mask_past = (
        ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:, :, : -(T - 1)]
        .cpu()
        .numpy()
    )
    padding_mask_fut = (
        ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:, :, -(T - 1) :]
        .cpu()
        .numpy()
    )
    y_pred = y_pred.reshape(B, N, T - 1, XYZ_3 // 3, 3)
    if not os.path.isdir(
        os.path.join(output_dir, "viz_results", "joint", f"batch_{batch_idx}")
    ):
        os.makedirs(
            os.path.join(output_dir, "viz_results", "joint", f"batch_{batch_idx}")
        )
    if not os.path.isdir(os.path.join(output_dir, "viz_results", "gifs")):
        os.makedirs(os.path.join(output_dir, "viz_results", "gifs"))

    # 遍历每个场景
    scene_idx_list = []
    for scene_idx in range(B):
        scene_idx_list.append(scene_idx)
        xy_mean = np.zeros((0, 2))
        if padding_mask_fut[scene_idx].sum() <= 1:
            continue

        # 遍历每一帧
        for frame_idx_ in range(TOTAL_T):
            plt.close()
            plt.cla()
            fig = plt.figure(figsize=(20, 9))
            ax = fig.add_subplot(111, projection="3d")

            # 设置3D图像属性
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            # 绘制轨迹和骨骼（只绘制过去轨迹） / Plot trajectory of agents (only past GT)
            frame_idx_past = min(frame_idx_, data.input_seq.shape[2] - 1)
            if padding_mask_past[scene_idx, :, frame_idx_past].sum() < 1:
                continue

            # 绘制历史轨迹点
            for agent_idx in range(x_gt.shape[1]):
                for traj_frame_idx in range(0, frame_idx_past + 1):
                    ax.scatter3D(
                        x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 0],
                        x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 1],
                        0,
                        c="black",
                        alpha=1.0,
                        s=5,
                    )

            # 根据时间分别处理历史和未来帧
            if frame_idx_ < data.input_seq.shape[2]:  # 历史帧
                if padding_mask_past[scene_idx, :, frame_idx_].sum() < 1:
                    continue
                for agent_idx in range(x_gt.shape[1]):
                    if padding_mask_past[scene_idx, agent_idx, frame_idx_] == False:
                        continue

                    # 计算平均位置
                    xy_mean = np.concatenate(
                        (
                            xy_mean,
                            np.expand_dims(
                                x_gt[scene_idx, agent_idx, frame_idx_, 0, :][:2], axis=0
                            ),
                        ),
                        axis=0,
                    )

                    # 绘制关节点
                    ax.scatter3D(
                        x_gt[scene_idx, agent_idx, frame_idx_, :, 0],
                        x_gt[scene_idx, agent_idx, frame_idx_, :, 1],
                        x_gt[scene_idx, agent_idx, frame_idx_, :, 2],
                        c="black",
                        alpha=1.0,
                        s=5,
                    )

                    # 绘制骨骼连接
                    if bones is None:
                        if x_gt.shape[3] == 15:
                            bones = TBI15_BONES
                        if x_gt.shape[3] == 13:
                            bones = EDGES_3DPW
                    for edge in bones:
                        x_ = [
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 0],
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 0],
                        ]
                        y_ = [
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 1],
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 1],
                        ]
                        z_ = [
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 2],
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 2],
                        ]
                        line = Line3D(x_, y_, z_, c="black")
                        ax.add_line(line)

                    # 绘制历史轨迹
                    for traj_frame_idx in range(0, frame_idx_ - 1):
                        ax.scatter3D(
                            x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 0],
                            x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 1],
                            0,
                            c="black",
                            alpha=1.0,
                            s=5,
                        )

            else:  # 未来帧
                frame_idx = frame_idx_ - data.input_seq.shape[2]
                if padding_mask_fut[scene_idx, :, frame_idx].sum() < 1:
                    continue

                # 绘制预测和真实轨迹/Plot pred/gt trajectory
                for agent_idx in range(y_pred.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False:
                        continue
                    for traj_frame_idx in range(0, data.output_seq.shape[2] - 1):
                        ax.scatter3D(
                            y_pred[scene_idx, agent_idx, traj_frame_idx, 0, 0],
                            y_pred[scene_idx, agent_idx, traj_frame_idx, 0, 1],
                            0,
                            c=HUMAN_COLORS[agent_idx],
                            alpha=1.0,
                            s=5,
                        )
                        ax.scatter3D(
                            y_gt[scene_idx, agent_idx, traj_frame_idx, 0, 0],
                            y_gt[scene_idx, agent_idx, traj_frame_idx, 0, 1],
                            0,
                            c="black",
                            alpha=1.0,
                            s=5,
                        )

                # 绘制预测的关节点和骨骼/plot pred motion
                for agent_idx in range(y_pred.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False:
                        continue
                    ax.scatter3D(
                        y_pred[scene_idx, agent_idx, frame_idx, :, 0],
                        y_pred[scene_idx, agent_idx, frame_idx, :, 1],
                        y_pred[scene_idx, agent_idx, frame_idx, :, 2],
                        c="black",
                        alpha=1.0,
                        s=5,
                    )

                    if bones is None:
                        bones = TBI15_BONES
                    for edge in bones:
                        x_ = [
                            y_pred[scene_idx, agent_idx, frame_idx, edge[0], 0],
                            y_pred[scene_idx, agent_idx, frame_idx, edge[1], 0],
                        ]
                        y_ = [
                            y_pred[scene_idx, agent_idx, frame_idx, edge[0], 1],
                            y_pred[scene_idx, agent_idx, frame_idx, edge[1], 1],
                        ]
                        z_ = [
                            y_pred[scene_idx, agent_idx, frame_idx, edge[0], 2],
                            y_pred[scene_idx, agent_idx, frame_idx, edge[1], 2],
                        ]
                        line = Line3D(x_, y_, z_, c=HUMAN_COLORS[agent_idx])
                        ax.add_line(line)

                # 绘制真实的关节点和骨骼 # plot GT motion
                for agent_idx in range(y_gt.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False:
                        continue
                    ax.scatter3D(
                        y_gt[scene_idx, agent_idx, frame_idx, :, 0],
                        y_gt[scene_idx, agent_idx, frame_idx, :, 1],
                        y_gt[scene_idx, agent_idx, frame_idx, :, 2],
                        c="black",
                        alpha=0.7,
                        s=5,
                    )

                    if bones is None:
                        bones = TBI15_BONES
                    for edge in bones:
                        x_ = [
                            y_gt[scene_idx, agent_idx, frame_idx, edge[0], 0],
                            y_gt[scene_idx, agent_idx, frame_idx, edge[1], 0],
                        ]
                        y_ = [
                            y_gt[scene_idx, agent_idx, frame_idx, edge[0], 1],
                            y_gt[scene_idx, agent_idx, frame_idx, edge[1], 1],
                        ]
                        z_ = [
                            y_gt[scene_idx, agent_idx, frame_idx, edge[0], 2],
                            y_gt[scene_idx, agent_idx, frame_idx, edge[1], 2],
                        ]
                        line = Line3D(x_, y_, z_, c="black", alpha=0.65)
                        ax.add_line(line)

            # 设置标题
            ax.set_title(f"frame: {frame_idx_}")

            # 计算场景中心和最大范围
            scene_center = [
                y_gt[scene_idx, :, :, :, 0].mean(),
                y_gt[scene_idx, :, :, :, 0].mean(),
            ]
            scene_max = np.max(
                (
                    np.abs(y_gt[scene_idx, :, :, :, 0]).max(),
                    np.abs(y_gt[scene_idx, :, :, :, 1]).max(),
                )
            )

            # 绘制水平面
            xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))
            z = (9 - xx - yy) * 0
            ax.plot_surface(xx, yy, z, color="0.5", alpha=0.06, zorder=0)

            # 绘制网格线
            for ii in range(-8, 10, 2):
                line_x = Line3D(
                    [scene_center[0] + ii, scene_center[0] + ii],
                    [scene_center[1] - 9, scene_center[1] + 9],
                    [0, 0],
                    c="grey",
                    alpha=0.5,
                    zorder=1,
                )
                ax.add_line(line_x)
            for jj in range(-8, 10, 2):
                line_x = Line3D(
                    [scene_center[0] - 9, scene_center[0] + 9],
                    [scene_center[1] + jj, scene_center[1] + jj],
                    [0, 0],
                    c="grey",
                    alpha=0.5,
                    zorder=1,
                )
                ax.add_line(line_x)

            # 设置坐标轴范围
            ax.set_xlim3d(
                [
                    scene_center[0] - (scene_max * 1.2),
                    scene_center[0] + (scene_max * 1.2),
                ]
            )
            ax.set_ylim3d(
                [
                    scene_center[1] - (scene_max * 1.2),
                    scene_center[1] + (scene_max * 1.2),
                ]
            )
            ax.set_zlim3d([0, 3])

            # 保存图像
            frame_save_name = (
                os.path.join(output_dir, "viz_results", "joint", f"batch_{batch_idx}")
                + f"/scene_{str(scene_idx).zfill(3)}_frame_{str(frame_idx_).zfill(3)}.png"
            )
            plt.savefig(frame_save_name, bbox_inches="tight", dpi=300)
            plt.close()
            plt.cla()
            sdf = 1

        # 生成GIF动画
        gif_save_dir = os.path.join(output_dir, "viz_results", "gifs")
        frame_save_dir = os.path.join(
            output_dir, "viz_results", "joint", f"batch_{batch_idx}"
        )
        save_as_gif_v2(gif_save_dir, frame_save_dir, TOTAL_T, batch_idx, scene_idx_list)


def viz_joint_jansang_v2(pred, gt, data, output_dir, batch_idx, bones=None):
    """
    可视化关节点预测结果(Jansang版本)
    Args:
        pred: 预测关节点 [B*N, T, 15, 3]
        gt: 真实关节点 [B*N, T, 15, 3]
        data: 数据对象
        output_dir: 输出目录
        batch_idx: 批次索引
        bones: 骨骼连接关系
    """
    # 创建输出目录
    if not os.path.isdir(os.path.join(output_dir, "viz_results", "joint_jansang")):
        os.makedirs(os.path.join(output_dir, "viz_results", "joint_jansang"))

    # 获取数据维度和处理数据
    B, N, T, XYZ_3 = data.output_seq.shape
    padding_mask_past = (
        ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:, :, : -(T - 1)]
        .cpu()
        .numpy()
    )
    padding_mask_fut = (
        ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:, :, -(T - 1) :]
        .cpu()
        .numpy()
    )
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3 // 3, 3)[:, :, :, 0].cpu().numpy()
    y_gt = gt.reshape(B, N, T - 1, XYZ_3 // 3, 3).cpu().numpy()
    y_pred = pred.reshape(B, N, T - 1, XYZ_3 // 3, 3).cpu().detach().numpy()

    # 转换数据类型
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    B, N, T, XYZ_3 = data.output_seq.shape
    # 获取总帧数
    TOTAL_T = data.input_seq.shape[2] + data.output_seq.shape[2] - 1
    xy_sizes = [4.5]
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3 // 3, 3).detach().cpu().numpy()
    y_gt = (
        data.output_seq.reshape(B, N, -1, XYZ_3 // 3, 3)
        .detach()
        .cpu()
        .numpy()[:, :, 1:]
    )
    padding_mask_past = (
        ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:, :, : -(T - 1)]
        .cpu()
        .numpy()
    )
    padding_mask_fut = (
        ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:, :, -(T - 1) :]
        .cpu()
        .numpy()
    )
    y_pred = y_pred.reshape(B, N, T - 1, XYZ_3 // 3, 3)
    if not os.path.isdir(
        os.path.join(output_dir, "viz_results", "joint_jansang", f"batch_{batch_idx}")
    ):
        os.makedirs(
            os.path.join(
                output_dir, "viz_results", "joint_jansang", f"batch_{batch_idx}"
            )
        )

    # 设置骨骼连接关系
    if bones is None:
        if x_gt.shape[3] == 15:
            bones = TBI15_BONES
        if x_gt.shape[3] == 13:
            bones = EDGES_3DPW

    # 遍历每个场景
    scene_idx_list = []
    for scene_idx in range(B):
        scene_idx_list.append(scene_idx)
        xy_mean = np.zeros((0, 2))
        if padding_mask_fut[scene_idx].sum() <= 1:
            continue

        # 遍历每一帧
        for frame_idx_ in range(0, TOTAL_T):
            plt.close()
            plt.cla()
            fig = plt.figure(figsize=(20, 9))
            ax = fig.add_subplot(111, projection="3d")

            # 设置3D图像属性
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            # 处理历史帧
            if frame_idx_ < data.input_seq.shape[2]:
                if padding_mask_past[scene_idx, :, frame_idx_].sum() < 1:
                    continue
                for agent_idx in range(x_gt.shape[1]):
                    if padding_mask_past[scene_idx, agent_idx, frame_idx_] == False:
                        continue

                    # 计算平均位置
                    xy_mean = np.concatenate(
                        (
                            xy_mean,
                            np.expand_dims(
                                x_gt[scene_idx, agent_idx, frame_idx_, 0, :][:2], axis=0
                            ),
                        ),
                        axis=0,
                    )

                    # 绘制关节点
                    ax.scatter3D(
                        x_gt[scene_idx, agent_idx, frame_idx_, :, 0],
                        x_gt[scene_idx, agent_idx, frame_idx_, :, 1],
                        x_gt[scene_idx, agent_idx, frame_idx_, :, 2],
                        c="black",
                        alpha=0.5,
                        s=5,
                    )

                    # 绘制骨骼连接
                    for edge in bones:
                        x_ = [
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 0],
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 0],
                        ]
                        y_ = [
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 1],
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 1],
                        ]
                        z_ = [
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 2],
                            x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 2],
                        ]
                        line = Line3D(x_, y_, z_, c="black")
                        ax.add_line(line)

                    # 绘制历史轨迹
                    loop_frame_idx = frame_idx_ - 1
                    max_loop = 10
                    loop_idx = 0
                    loop_alpha = np.linspace(0.75, 0, np.min((max_loop, frame_idx_)))
                    while loop_idx < 10 and loop_frame_idx >= 0:
                        if (
                            padding_mask_fut[scene_idx, agent_idx, loop_frame_idx]
                            == False
                        ):
                            continue
                        ax.scatter3D(
                            x_gt[scene_idx, agent_idx, loop_frame_idx, :, 0],
                            x_gt[scene_idx, agent_idx, loop_frame_idx, :, 1],
                            x_gt[scene_idx, agent_idx, loop_frame_idx, :, 2],
                            c="black",
                            alpha=loop_alpha[loop_idx],
                            s=5,
                        )
                        for edge in bones:
                            x_ = [
                                x_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 0],
                                x_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 0],
                            ]
                            y_ = [
                                x_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 1],
                                x_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 1],
                            ]
                            z_ = [
                                x_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 2],
                                x_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 2],
                            ]
                            line = Line3D(
                                x_, y_, z_, c="black", alpha=loop_alpha[loop_idx]
                            )
                            ax.add_line(line)
                        loop_frame_idx -= 1
                        loop_idx += 1

            # 处理未来帧
            else:
                frame_idx = frame_idx_ - data.input_seq.shape[2]
                if padding_mask_fut[scene_idx, :, frame_idx].sum() < 1:
                    continue

                # 绘制真实关节点和骨骼
                for agent_idx in range(y_gt.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False:
                        continue
                    ax.scatter3D(
                        y_gt[scene_idx, agent_idx, frame_idx, :, 0],
                        y_gt[scene_idx, agent_idx, frame_idx, :, 1],
                        y_gt[scene_idx, agent_idx, frame_idx, :, 2],
                        c="black",
                        alpha=1,
                        s=5,
                    )

                    for edge in bones:
                        x_ = [
                            y_gt[scene_idx, agent_idx, frame_idx, edge[0], 0],
                            y_gt[scene_idx, agent_idx, frame_idx, edge[1], 0],
                        ]
                        y_ = [
                            y_gt[scene_idx, agent_idx, frame_idx, edge[0], 1],
                            y_gt[scene_idx, agent_idx, frame_idx, edge[1], 1],
                        ]
                        z_ = [
                            y_gt[scene_idx, agent_idx, frame_idx, edge[0], 2],
                            y_gt[scene_idx, agent_idx, frame_idx, edge[1], 2],
                        ]
                        line = Line3D(x_, y_, z_, c="black")
                        ax.add_line(line)

                    # 绘制历史轨迹
                    loop_frame_idx = frame_idx - 1
                    max_loop = 10
                    loop_idx = 0
                    loop_alpha = np.linspace(0.75, 0, np.min((max_loop, frame_idx)))
                    while loop_idx < 10 and loop_frame_idx >= 0:
                        if (
                            padding_mask_fut[scene_idx, agent_idx, loop_frame_idx]
                            == False
                        ):
                            continue
                        ax.scatter3D(
                            y_gt[scene_idx, agent_idx, loop_frame_idx, :, 0],
                            y_gt[scene_idx, agent_idx, loop_frame_idx, :, 1],
                            y_gt[scene_idx, agent_idx, loop_frame_idx, :, 2],
                            c="black",
                            alpha=loop_alpha[loop_idx],
                            s=5,
                        )
                        for edge in bones:
                            x_ = [
                                y_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 0],
                                y_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 0],
                            ]
                            y_ = [
                                y_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 1],
                                y_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 1],
                            ]
                            z_ = [
                                y_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 2],
                                y_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 2],
                            ]
                            line = Line3D(
                                x_, y_, z_, c="black", alpha=loop_alpha[loop_idx]
                            )
                            ax.add_line(line)
                        loop_frame_idx -= 1
                        loop_idx += 1

            ax.set_title(f"frame: {frame_idx_}")

            scene_center = [
                y_gt[scene_idx, :, :, :, 0].mean(),
                y_gt[scene_idx, :, :, :, 0].mean(),
            ]
            scene_max = np.max(
                (
                    np.abs(y_gt[scene_idx, :, :, :, 0]).max(),
                    np.abs(y_gt[scene_idx, :, :, :, 1]).max(),
                )
            )
            # Plot horizontal plane
            xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))
            z = (9 - xx - yy) * 0
            ax.plot_surface(xx, yy, z, color="0.5", alpha=0.06, zorder=0)

            # Plot xy grid lines
            for ii in range(-8, 10, 2):
                line_x = Line3D(
                    [scene_center[0] + ii, scene_center[0] + ii],
                    [scene_center[1] - 9, scene_center[1] + 9],
                    [0, 0],
                    c="grey",
                    alpha=0.5,
                    zorder=1,
                )
                ax.add_line(line_x)
            for jj in range(-8, 10, 2):
                line_x = Line3D(
                    [scene_center[0] - 9, scene_center[0] + 9],
                    [scene_center[1] + jj, scene_center[1] + jj],
                    [0, 0],
                    c="grey",
                    alpha=0.5,
                    zorder=1,
                )
                ax.add_line(line_x)
            ax.set_xlim3d(
                [
                    scene_center[0] - (scene_max * 1.2),
                    scene_center[0] + (scene_max * 1.2),
                ]
            )
            ax.set_ylim3d(
                [
                    scene_center[1] - (scene_max * 1.2),
                    scene_center[1] + (scene_max * 1.2),
                ]
            )
            ax.set_zlim3d([0, 3])
            frame_save_name = (
                os.path.join(
                    output_dir, "viz_results", "joint_jansang", f"batch_{batch_idx}"
                )
                + f"/scene_{str(scene_idx).zfill(3)}_frame_{str(frame_idx_).zfill(3)}.png"
            )
            plt.savefig(frame_save_name, bbox_inches="tight", dpi=300)
            plt.close()
            plt.cla()


def save_as_gif_v2(gif_save_dir, frame_save_dir, gifFrames, batch_idx, scene_idx_list):
    """将PNG图片序列保存为GIF动画
    Args:
        gif_save_dir: GIF保存目录
        frame_save_dir: 帧图片目录
        gifFrames: 每个GIF包含的帧数
        batch_idx: 批次索引
        scene_idx_list: 场景索引列表
    """
    # 获取所有PNG图片并排序
    imgs = glob.glob(frame_save_dir + "/*.png")
    imgs = sorted(imgs)
    frames_ = []

    # 读取所有图片并保存到列表中
    for img in imgs:
        temp = Image.open(img)
        image = temp.copy()
        frames_.append(image)
        temp.close()

    frames = frames_

    # 如果未指定帧数,使用所有帧
    if gifFrames is None:
        gifFrames = len(frames)

    # 按指定帧数分批生成GIF
    for batchIdx in range(len(frames) // gifFrames):
        # 生成GIF文件名
        gifFilename = f"batch_{batch_idx}_frames_{batchIdx*gifFrames}_{(batchIdx+1)*gifFrames}.gif"

        # 保存GIF动画
        frames[0 + int(batchIdx * gifFrames)].save(
            os.path.join(gif_save_dir, gifFilename),
            save_all=True,
            append_images=frames[
                (batchIdx * gifFrames) + 1 : (batchIdx + 1) * gifFrames
            ],
            optimize=False,
            duration=100,  # 帧间隔100ms
            loop=0,  # 循环播放
        )
