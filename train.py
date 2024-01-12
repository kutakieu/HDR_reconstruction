from pathlib import Path

import hydra
import wandb
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from reconsthdr import Env
from reconsthdr.dataset.dataset_factory import DatasetFactory
from reconsthdr.lightning_wrapper import LightningHdrEstimator
from reconsthdr.metrics import metrics_dict
from reconsthdr.utils.remote_logger import WandbLoggerUtils


@hydra.main(
    config_path=Env().config_dir,
    config_name=Env().config_name,
    version_base=None,
)
def main(cfg: DictConfig):
    dataset_factory = DatasetFactory(cfg)
    train_dataset = dataset_factory.create_dataset("train")
    val_dataset = dataset_factory.create_dataset("val")
    test_dataset = dataset_factory.create_dataset("test")

    hdr_estimator = LightningHdrEstimator(cfg)

    wandb_logger = None
    if Env().wandb_api_key:
        wandb.login(key=Env().wandb_api_key)
        wandb_logger = WandbLogger(
            project=cfg.model.name, 
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            )
        
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator.device,
        val_check_interval=1.0,
        devices=cfg.training.accelerator.gpus,
        log_every_n_steps=cfg.training.log_every_n_batch,
        max_epochs=cfg.training.n_epochs,
        callbacks=make_callbacks(cfg),
        logger=wandb_logger,
    )

    trainer.fit(
        model=hdr_estimator,
        train_dataloaders=train_dataset.create_dataloader(),
        val_dataloaders=val_dataset.create_dataloader(),
    )
    trainer.test(model=hdr_estimator, dataloaders=test_dataset.create_dataloader(), ckpt_path="best")

    if wandb_logger is not None:
        wandb_logger_utils = WandbLoggerUtils(wandb_logger)
        wandb_logger_utils.log_data_split(train_dataset.data_samples, "train")
        wandb_logger_utils.log_data_split(val_dataset.data_samples, "val")
        wandb_logger_utils.log_data_split(test_dataset.data_samples, "test")

        checkpoint_files = list(Path(trainer.checkpoint_callback.dirpath).glob("*.ckpt")) if trainer.checkpoint_callback else []
        for checkpoint_file in checkpoint_files:
            wandb_logger_utils.log_model(checkpoint_file, hdr_estimator)


def make_callbacks(cfg: DictConfig):
    early_stop_callback = EarlyStopping(
        monitor="loss_val",
        min_delta=0.01,
        patience=cfg.training.early_stopping_patience,
        verbose=False,
        mode="min",
    )
    best_val_loss_checkpoint_callback = ModelCheckpoint(
        monitor='loss_val',
        filename='best-val-loss-epoch{epoch:02d}',
        auto_insert_metric_name=False,
        save_weights_only=False,
    )
    best_val_accuracy_checkpoint_callbacks = []
    for key in metrics_dict.keys():
        best_val_accuracy_checkpoint_callbacks.append(
            ModelCheckpoint(
                monitor=f'{key}_val',
                filename=f'best-val-{key}'+'-epoch{epoch:02d}',
                auto_insert_metric_name=False,
                save_weights_only=False,
            )
        )
    return [early_stop_callback, best_val_loss_checkpoint_callback] + best_val_accuracy_checkpoint_callbacks


if __name__ == "__main__":
    main()
