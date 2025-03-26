import torch

# 加载检查点
model_ckpt = torch.load("D:\\Downloads\\T2P\\T2P-main\\outputs\\cmu_t2p\\2025-03-23\\21-19-40_debug\\checkpoints\\epoch=59-step=12240.ckpt")

# 打印检查点内容
print(model_ckpt.keys())
print(model_ckpt['state_dict'].keys())
