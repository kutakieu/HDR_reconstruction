from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from torchvision.transforms.v2 import Compose, Normalize, ToTensor

from . import BaseDataset
from .augmentation import apply_hue_jitter, flip, random_rotate
from .tone_mappers import (BaseToneMapper, Drago, Durand, Mantiuk,
                           PercentileExposure, Reinhard)
from .utils import load_hdr

ToneMappers = [
    PercentileExposure,
    Reinhard,
    Mantiuk,
    Drago,
    Durand,
]

@dataclass
class HdrDataSample:
    hdr_file: Path
    tonemapper: BaseToneMapper

class PanoHdrDataset(BaseDataset):
    def __init__(self, data_dir: Path, file_formats: List[str]=["hdr", "exr"], is_training: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.is_training = is_training
        self.data_dir = data_dir
        self.data_samples: List[HdrDataSample] = []
        for fmt in file_formats:
            for hdr_file in self.data_dir.glob(f"*.{fmt}"):
                for tm in ToneMappers:
                    self.data_samples.append(HdrDataSample(hdr_file, tm))

        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        hdr_img = load_hdr(self.data_samples[idx].hdr_file)
        if self.is_training:
            hdr_img = self.augment(hdr_img)
        ldr_img = self.data_samples[idx].tonemapper(randomize=True)(hdr_img)
        return ToTensor(hdr_img), self.img_transform(ldr_img)
    
    def augment(self, hdr_img: np.ndarray):
        if self.flip and np.random.rand() < 0.5:
            hdr_img = flip(hdr_img)
        if self.rotate:
            hdr_img = random_rotate(hdr_img)
        if self.color_jitter:
            hdr_img = apply_hue_jitter(hdr_img)
        return hdr_img
