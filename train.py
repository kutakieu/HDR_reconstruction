import hydra
import wandb
from lightning import pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from reconsthdr import Env
from reconsthdr.dataset.dataset_factory import DatasetFactory
from reconsthdr.lightning_wrapper import LightningHdrEstimator
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

    early_stop_callback = EarlyStopping(
        monitor="loss_val",
        min_delta=0.01,
        patience=cfg.training.early_stopping_patience,
        verbose=False,
        mode="min",
    )

    if Env().wandb_api_key:
        wandb.login(key=Env().wandb_api_key)
        wandb_logger = WandbLogger(
            project=cfg.model.name, 
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )
        wandb_logger_utils = WandbLoggerUtils(wandb_logger)
        wandb_logger_utils.log_data_split(train_dataset.data_samples, "train")
        wandb_logger_utils.log_data_split(val_dataset.data_samples, "val")
        wandb_logger_utils.log_data_split(test_dataset.data_samples, "test")


    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator.device,
        val_check_interval=1.0,
        devices=cfg.training.accelerator.gpus,
        log_every_n_steps=cfg.training.log_every_n_batch,
        max_epochs=cfg.training.n_epochs,
        callbacks=[early_stop_callback],
        logger=wandb_logger if Env().wandb_api_key else None,
    )

    trainer.fit(
        model=hdr_estimator,
        train_dataloaders=train_dataset.create_dataloader(),
        val_dataloaders=val_dataset.create_dataloader(),
    )

    trainer.test(model=hdr_estimator, dataloaders=test_dataset.create_dataloader())


if __name__ == "__main__":
    main()
