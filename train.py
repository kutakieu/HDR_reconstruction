import hydra
import wandb
from lightning import pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from reconsthdr import Env
from reconsthdr.dataset.dataset_factory import DatasetFactory
from reconsthdr.lightning_wrapper import LightningHdrEstimator


@hydra.main(
    config_path=Env().config_dir,
    config_name=Env().config_name,
    version_base=None,
)
def main(cfg: DictConfig):
    dataset_factory = DatasetFactory(cfg)
    train_dataloader = dataset_factory.create_dataset("train").create_dataloader()
    val_dataloader = dataset_factory.create_dataset("val").create_dataloader()
    test_dataloader = dataset_factory.create_dataset("test").create_dataloader()

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
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    trainer.test(model=hdr_estimator, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
