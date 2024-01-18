from pathlib import Path
from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms.v2 import Compose, Normalize, ToTensor

from reconsthdr.dataset.tone_mappers import calibrate_hdr
from reconsthdr.models import model_factory


class Predictor:
    def __init__(self, cfg: Union[str, DictConfig], weight_file: Union[str, Path], device="cpu") -> None:
        if isinstance(cfg, str):
            cfg = OmegaConf.load(cfg)
        self.device = device
        self.net = model_factory(cfg)
        self.net.to(self.device)
        if Path(weight_file).suffix == ".ckpt":
            from reconsthdr.lightning_wrapper import LightningHdrEstimator
            checkpoint = torch.load(weight_file)
            lightning_model = LightningHdrEstimator(cfg)
            lightning_model.load_state_dict(checkpoint['state_dict'])
            self.net = lightning_model.net
        else:
            self.net.load_state_dict(torch.load(weight_file)["state_dict"])
        self.net.eval()

        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, ldr: np.ndarray) -> np.ndarray:
        ldr_tensor = self.img_transform(ldr)
        with torch.no_grad():
            ldr_tensor = ldr_tensor.unsqueeze(0).to(self.device)
            hdr = np.exp(self.net(ldr_tensor).cpu().permute(0, 2, 3, 1).numpy()[0])
            return calibrate_hdr(hdr, ldr)
