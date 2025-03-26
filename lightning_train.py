'''
训练模型
主要功能:
1. 加载模型配置和权重
2. 准备训练数据
3. 运行训练并保存检查点
'''
# 导入必要的库
import os
import yaml
import hydra
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from hydra.core.hydra_config import HydraConfig

# 导入PyTorch Lightning相关组件
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


# 使用hydra管理配置,指定配置文件路径和名称
@hydra.main(version_base=None, config_path="./conf/", config_name="train_config_3dpw_t2p") 
def main(conf):
    # 设置随机种子
    pl.seed_everything(conf.seed)
    # 获取输出目录
    output_dir = HydraConfig.get().runtime.output_dir

    # 实例化模型
    model = instantiate(conf.model.target)
    model.output_dir = output_dir
    model.net = instantiate(conf.net.target)
    
    # 如果指定了checkpoint,加载预训练模型
    if conf.checkpoint is not None:
        print(f"Loading model from {conf.checkpoint}...")
        checkpoint = to_absolute_path(conf.checkpoint)
        assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"
    
        model_ckpt = torch.load(checkpoint)
        model.load_state_dict(model_ckpt['state_dict'])
    
    # 将模型移至GPU
    model.net.cuda()
    
    # 设置TensorBoard日志记录器
    logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    
    # 设置模型检查点保存
    model_checkpoint = ModelCheckpoint(dirpath=os.path.join(output_dir, "checkpoints"), monitor=conf.monitor, save_top_k=conf.save_top_k, mode='min')
    
    # 配置训练器
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=conf.epochs,
        callbacks=[model_checkpoint],
        limit_val_batches=conf.limit_val_batches,
        limit_test_batches=conf.limit_test_batches,
    )

    # 准备数据
    datamodule: pl.LightningDataModule = instantiate(conf.datamodule)
    datamodule.setup()
    
    # 开始训练
    print('Start training')
    # trainer.validate(model, datamodule.val_dataloader()) 
    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())
    ff = 1


if __name__ == "__main__":
    # 设置多进程启动方式和矩阵计算精度
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    main()