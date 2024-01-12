import tempfile
from abc import ABC
from pathlib import Path
from typing import List, Literal, Union

import torch
import wandb
from lightning import pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from ..dataset import DataSample

SplitType = Literal["train", "val", "test"]


class BaseLoggerUtils(ABC):
    def log_any(self, data: dict):
        raise NotImplementedError

    def log_artifact(self, local_path: Union[str, Path], type: str, description: str) -> None:
        raise NotImplementedError

    def log_config_file(self, cfg):
        with tempfile.TemporaryDirectory() as dname:
            config_file_path = Path(dname) / "config.yaml"
            OmegaConf.save(config=cfg, f=config_file_path)
            self.log_artifact(config_file_path, "config_file")

    def log_str_as_file(
        self,
        file_content: str,
        filename: Union[str, Path],
        data_type: str,
        description: str="",
    ):
        with tempfile.TemporaryDirectory() as dname:
            local_tmp_filepath = Path(dname) / filename
            with open(local_tmp_filepath, "w") as f:
                f.write(file_content)
            self.log_artifact(local_tmp_filepath, type=data_type, description=description)

    def log_data_split(self, data_samples: List[DataSample], split_type: SplitType):
        data_sample_list_as_str = "\n".join([str(data_sample.hdr_file) for data_sample in data_samples])
        self.log_str_as_file(data_sample_list_as_str, f"{split_type}.csv", "splits")

    def log_model(self, checkpoint_path: Union[str, Path], lightning_model: pl.LightningModule):
        checkpoint = torch.load(checkpoint_path)
        lightning_model.load_state_dict(checkpoint['state_dict'])

        with tempfile.TemporaryDirectory() as dname:
            model_path = Path(dname) / f"{Path(checkpoint_path).stem}.pth"
            torch.save({"state_dict": lightning_model.net.state_dict()}, model_path)
            self.log_artifact(model_path, "model", "")


class WandbLoggerUtils(BaseLoggerUtils):
    def __init__(self, logger: WandbLogger):
        self.logger = logger

    def log_any(self, data: dict):
        self.logger.experiment.log(data)

    def log_artifact(self, local_path: Union[str, Path], type: str, description: str) -> None:
        local_path = Path(local_path)
        artifact = wandb.Artifact(f"{wandb.run.id}_{local_path.stem}", type=type, description=description)
        if local_path.is_dir():
            artifact.add_dir(local_path)
        else:
            artifact.add_file(local_path)
        self.logger.experiment.log_artifact(artifact)
