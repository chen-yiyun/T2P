# 读取3dpw里的一个.pt文件(preprocessed\3dpw_input10_v2\train\0.pt),并将读取到的TemporalData对象转化为字典
import numpy as np
import torch
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams["font.family"] = "SimHei"  # 使用黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 指定文件路径
pt_path = osp.join("preprocessed", "3dpw_input10_v2", "hip", "train", "0.pt")

# 读取.pt文件
data = torch.load(pt_path)

# 将TemporalData对象转换为字典
data_dict = dict(data)
body_xyz = data_dict["body_xyz"]

body_xyz_one_f_t = data_dict["body_xyz"][0][
    0
]  # 一个智能体在一个时间步的关节点坐标，共13个关节点，每个关节点有3个坐标(xyz)

# 可视化body_xyz_one_f_t
plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")

# 绘制关节点
ax.scatter3D(
    body_xyz_one_f_t[:, 0],  # x坐标
    body_xyz_one_f_t[:, 1],  # y坐标
    body_xyz_one_f_t[:, 2],  # z坐标
    c="black",
    alpha=1.0,
    s=50,
)

# 标上每个关节点的索引
for i in range(body_xyz_one_f_t.shape[0]):
    ax.text(
        body_xyz_one_f_t[i, 0],
        body_xyz_one_f_t[i, 1],
        body_xyz_one_f_t[i, 2],
        str(i),  # 关节点索引
        color="red",
    )

# 定义3DPW数据集骨骼连接关系
EDGES_3DPW = np.array(
    [
        (0, 1),  # 连接关节0和关节1
        (1, 8),  # 连接关节1和关节8
        (8, 7),  # 连接关节8和关节7
        (7, 0),  # 连接关节7和关节0
        (0, 2),  # 连接关节0和关节2
        (2, 4),  # 连接关节2和关节4
        (1, 3),  # 连接关节1和关节3
        (3, 5),  # 连接关节3和关节5
        (7, 9),  # 连接关节7和关节9
        (9, 11),  # 连接关节9和关节11
        (8, 10),  # 连接关节8和关节10
        (10, 12),  # 连接关节10和关节12
        (6, 7),  # 连接关节6和关节7
        (6, 8),  # 连接关节6和关节8
    ]
)

# 绘制骨骼连接
for edge in EDGES_3DPW:
    x = [body_xyz_one_f_t[edge[0], 0], body_xyz_one_f_t[edge[1], 0]]  # x坐标
    y = [body_xyz_one_f_t[edge[0], 1], body_xyz_one_f_t[edge[1], 1]]  # y坐标
    z = [body_xyz_one_f_t[edge[0], 2], body_xyz_one_f_t[edge[1], 2]]  # z坐标
    ax.plot(x, y, z, c="black")  # 绘制连接线

# 设置坐标轴标签
ax.set_xlabel("X")  # X轴标签
ax.set_ylabel("Y")  # Y轴标签
ax.set_zlabel("Z")  # Z轴标签

# 设置视角
# 设置视角，调整3D图的观察角度
# elev参数控制仰角，azim参数控制方位角
ax.view_init(elev=20, azim=45)

# 设置坐标轴范围
margin = 0.5
ax.set_xlim(
    [
        body_xyz_one_f_t[:, 0].min() - margin,
        body_xyz_one_f_t[:, 0].max() + margin,
    ]  # X轴范围
)
ax.set_ylim(
    [
        body_xyz_one_f_t[:, 1].min() - margin,
        body_xyz_one_f_t[:, 1].max() + margin,
    ]  # Y轴范围
)
ax.set_zlim(
    [
        body_xyz_one_f_t[:, 2].min() - margin,
        body_xyz_one_f_t[:, 2].max() + margin,
    ]  # Z轴范围
)

# 1.髋部
hip_xyz = data_dict["body_xyz"][0][0][0]  # 获取髋部坐标
ax.scatter3D(
    hip_xyz[0],
    hip_xyz[1],
    hip_xyz[2],
    c="blue",  # 髋部颜色
    s=100,  # 髋部大小
)
print("hip_xyz:", hip_xyz)  # 打印髋部坐标

# 2.计算所有关节点坐标的平均值
mean_xyz = body_xyz.mean(dim=2)[0][0]  # 计算平均坐标
ax.scatter3D(
    mean_xyz[0],
    mean_xyz[1],
    mean_xyz[2],
    c="red",  # 平均值颜色
    s=100,  # 平均值大小
)
print("mean_xyz:", mean_xyz)  # 打印平均坐标

# 3.计算加权平均位置
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
)  # 创建权重张量
weights /= weights.sum()  # 归一化权重
weighted_xyz = torch.tensordot(body_xyz, weights, dims=([2], [0]))[0][
    0
]  # 计算加权平均坐标

# 在图中标上加权平均值
ax.scatter3D(
    weighted_xyz[0],
    weighted_xyz[1],
    weighted_xyz[2],
    c="yellow",  # 加权平均值颜色
    s=100,  # 加权平均值大小
)
print("weighted_xyz:", weighted_xyz)  # 打印加权平均坐标

plt.title("3D Human Pose Visualization")  # 设置图标题

plt.title("3D 人体姿态可视化")  # 设置图标题
hip_patch = plt.Line2D(
    [0], [0], marker="o", color="w", label="髋部", markerfacecolor="blue", markersize=10
)
mean_patch = plt.Line2D(
    [0],
    [0],
    marker="o",
    color="w",
    label="中心点",
    markerfacecolor="red",
    markersize=10,
)
weighted_patch = plt.Line2D(
    [0],
    [0],
    marker="o",
    color="w",
    label="加权平均点",
    markerfacecolor="yellow",
    markersize=10,
)
plt.legend(
    handles=[hip_patch, mean_patch, weighted_patch], loc="upper left"
)  # 添加图例
plt.show()  # 显示图形
plt.savefig("3dpw_hip_mean.png")  # 保存图形（可选）

