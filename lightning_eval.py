'''
模型评估脚本
主要功能:
1. 加载模型配置和权重
2. 准备验证数据
3. 运行验证并保存结果
'''

# 导入必要的库
import os
import yaml
import hydra
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from hydra.core.hydra_config import HydraConfig
import json

# 使用hydra管理配置,指定配置文件路径和名称
@hydra.main(version_base=None, config_path="./conf/", config_name="eval_config_3dpw_t2p")
def main(conf):
    # 设置随机种子
    pl.seed_everything(conf.seed)
    # 获取输出目录
    output_dir = HydraConfig.get().runtime.output_dir

    # 实例化模型
    model = instantiate(conf.model.target)
    model.output_dir = output_dir
    model.net = instantiate(conf.net.target)

    # 加载检查点
    checkpoint = to_absolute_path(conf.checkpoint)
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    # 根据检查点格式加载模型权重
    if checkpoint[-5:] == '.ckpt':
        # 处理.ckpt格式的检查点
        ckpt_file = torch.load(checkpoint)
        ckpt_state_dict = ckpt_file['state_dict']
        # 处理权重字典的键名
        for key in list(ckpt_state_dict.keys()):
            ckpt_state_dict[key[4:]] = ckpt_state_dict[key]
            del ckpt_state_dict[key]
        model.net.load_state_dict(ckpt_state_dict)
        print('Model loaded!')
    else:
        # 处理其他格式的检查点
        model_ckpt = torch.load(checkpoint)
        model.net.load_state_dict(model_ckpt['model'])
        print('Model loaded!')
        
    # 重置验证指标
    model.val_metrics.reset()

    # 配置训练器
    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=1,
        limit_val_batches=conf.limit_val_batches,
        limit_test_batches=conf.limit_test_batches,
    )

    # datamodule: pl.LightningDataModule = instantiate(conf.datamodule, test=conf.test)
    # 准备数据
    datamodule: pl.LightningDataModule = instantiate(conf.datamodule)
    datamodule.setup()
    
    # 开始验证
    print('Start validation')
    val_results = trainer.validate(model, datamodule.val_dataloader())
    # 保存验证结果
    with open(output_dir+'\eval_dir.json','w') as f:
        json.dump(val_results[0], f, indent=4)
    ff = 1


if __name__ == "__main__":
    # 设置多进程启动方式和矩阵计算精度
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    main()
