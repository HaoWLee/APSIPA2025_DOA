import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from apsipa25_dataset import DOADataset_10_stft
import wandb
import json
from crnn_model import CRNNModel
import argparse
from apsipa25_dataset_weight import compute_class_weights_from_json

class DOALitModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = CRNNModel(self.config.input_shape, self.config.n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.config.n_classes)
        self.lr = self.config.lr
        self.test_loss = 0.0
        self.test_acc_value = 0.0
        self.class_weights = compute_class_weights_from_json("/home/lihaowen/BEWO/BEWO_SS_Annotation_v1/BEWO_SS_Annotation_v1/audiocaps_single_train.jsonl")
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
    def forward(self, ipd, ild):
        return self.model(ipd, ild)

    def training_step(self, batch, batch_idx):
        stft_cat, ild, label = batch
        logits = self(stft_cat, ild)
        #loss = self.criterion(logits, label)
        loss = F.cross_entropy(logits, label, weight=self.class_weights.to(logits.device))
        acc = self.accuracy(logits, label)
        self.log("train/loss", loss,on_step = True, on_epoch=True,prog_bar=True)
        self.log("train/acc", acc,on_step = True,on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        stft_cat, ild, label = batch
        logits = self(stft_cat, ild)
        loss = self.criterion(logits, label)
        acc = self.accuracy(logits, label)
        self.log("val/loss", loss, on_epoch=True,prog_bar=True)
        self.log("val/acc", acc,on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        stft_cat, ild, label = batch
        logits = self(stft_cat, ild)
        loss = self.criterion(logits, label)
        acc = self.accuracy(logits, label)
        self.log("test/loss", loss,on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train(config):
    import os
    from datetime import datetime

    # ==== 自动生成唯一实验名称 ====
    exp_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_name = f"{config.experiment_name}"
    output_dir = os.path.join("outputs", unique_name)
    os.makedirs(output_dir, exist_ok=True)

    # ==== 保存当前 config ====
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    # ==== 初始化 wandb ====
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="doa_apsipa",
        tags=["2025_lhw"],
        config=config,
        name=unique_name
    )
    train_set = DOADataset_10_stft(jsonl_path="/home/lihaowen/BEWO/BEWO_SS_Annotation_v1/BEWO_SS_Annotation_v1/audiocaps_single_train.jsonl",
                           audio_root="/home/lihaowen/BEWO/BEWO_SS_Audio_v1/BEWO_SS_Audio_v1/audiocaps_single_train")
    val_set = DOADataset_10_stft(jsonl_path="/home/lihaowen/BEWO/BEWO_SS_Annotation_v1/BEWO_SS_Annotation_v1/audiocaps_single_val.jsonl",
                         audio_root="/home/lihaowen/BEWO/BEWO_SS_Audio_v1/BEWO_SS_Audio_v1/audiocaps_single_val")
    test_set = DOADataset_10_stft(
        jsonl_path="/home/lihaowen/BEWO/BEWO_SS_Annotation_v1/BEWO_SS_Annotation_v1/audiocaps_single_test.jsonl",
        audio_root="/home/lihaowen/BEWO/BEWO_SS_Audio_v1/BEWO_SS_Audio_v1/audiocaps_single_test")
    train_dl = DataLoader(
        dataset=train_set,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_dl = DataLoader(
        dataset=val_set,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )
    test_dl = DataLoader(
        dataset=test_set,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )
    # ==== 初始化模型 ====
    pl_module = DOALitModule(config)

    # ==== 计算模型复杂度并记录到 wandb ====

    from pathlib import Path

    base_dir = Path(__file__).parent.resolve()
    output_dir = base_dir / "outputs" / unique_name
    # ==== 训练器 Trainer ====
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val/acc",
        mode="max",
        patience=10,
        verbose=True
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(output_dir, "checkpoints"),
                filename="{epoch}-{val/acc:.4f}",
                save_last=True,
                save_top_k=1,
                monitor="val/acc",
                mode="max"
            )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        accelerator='gpu',
        devices=1,
        precision=config.precision,
        callbacks=[checkpoint_callback,early_stop_callback]
    )
    # ==== 开始训练 ====
    trainer.fit(pl_module, train_dl, val_dl)

    # ==== 最终测试 ====
    #trainer.test(ckpt_path='last', dataloaders=test_dl)
    # 1. 使用 best checkpoint 测试
    print(">>> Testing with best checkpoint")
    trainer.test(pl_module, dataloaders=test_dl, ckpt_path=checkpoint_callback.best_model_path)
    pl_module.logger.experiment.log({"test/best_ckpt_path": checkpoint_callback.best_model_path,"test_best/acc":pl_module.test_acc_value,
                                     "test_best/loss":pl_module.test_loss})

    # 2. 使用 last checkpoint 测试
    print(">>> Testing with last checkpoint")
    trainer.test(pl_module, dataloaders=test_dl, ckpt_path='last')
    pl_module.logger.experiment.log({"test/last_ckpt_path": 'last',"test_last/acc":pl_module.test_acc_value,
                                     "test_last/loss":pl_module.test_loss})
    # ==== 关闭 wandb ====

    # ==== 关闭 wandb ====
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DOA_BEWO')

    # general
    parser.add_argument('--project_name', type=str, default="DOA_APSIPA")
    parser.add_argument('--experiment_name', type=str, default="CRNN_stft_10_besttest_wLoss")
    parser.add_argument('--num_workers', type=int, default=12)  # number of workers for dataloaders
    parser.add_argument('--precision', type=str, default="32")

    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, default=None)  # for loading trained model, corresponds to wandb id

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    # peak learning rate (in cosinge schedule)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    parser.add_argument('--input_shape', nargs=3, type=int, default=[4, 257, 128])
    parser.add_argument('--n_classes', type=int, default=37)

    args = parser.parse_args()
    train(args)
