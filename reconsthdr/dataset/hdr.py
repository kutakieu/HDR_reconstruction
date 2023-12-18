from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from torchvision.transforms import ToTensor

from . import BaseDataset
from .augmentation import flip, random_rotate
from .tone_mappers import (BaseTMO, Drago, Durand, Mantiuk, PercentileExposure,
                           Reinhard)

TMOs = [
    PercentileExposure,
    Reinhard,
    Mantiuk,
    Drago,
    Durand,
]

@dataclass
class HdrDataSample:
    hdr_file: Path
    tmo: BaseTMO

class PanoHdrDataset(BaseDataset):
    def __init__(self, data_dir: Path, file_formats: List[str]=["hdr", "exr"], is_training: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.is_training = is_training
        self.data_dir = data_dir
        self.data_samples: List[HdrDataSample] = []
        for fmt in file_formats:
            for hdr_file in self.data_dir.glob(f"*.{fmt}"):
                for tmo in TMOs:
                    self.data_samples.append(HdrDataSample(hdr_file, tmo))
    
    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        hdr_img = cv2.cvtColor(cv2.imread(
            self.data_samples[idx].hdr_file, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
        ), cv2.COLOR_BGR2RGB)
        if self.is_training:
            hdr_img = self.augment(hdr_img)
        ldr_img = self.data_samples[idx].tmo(hdr_img)
        return ToTensor(hdr_img), ToTensor(ldr_img)
    
    def augment(self, hdr_img: np.ndarray):
        if self.flip and np.random.rand() < 0.5:
            hdr_img = flip(hdr_img)
        if self.rotate:
            hdr_img = random_rotate(hdr_img)
        return hdr_img


