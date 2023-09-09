from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tone_mappers import (BaseTMO, Drago, Durand, Mantiuk, PercentileExposure,
                          Reinhard)

from . import BaseDataset

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

class HdrDataset(BaseDataset):
    def __init__(self, data_dir: Path, file_formats: List[str]=["hdr", "exr"], **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.data_samples: List[HdrDataSample] = []
        for fmt in file_formats:
            for hdr_file in self.data_dir.glob(f"*.{fmt}"):
                for tmo in TMOs:
                    self.data_samples.append(HdrDataSample(hdr_file, tmo))
    
    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        hdr_img_raw = cv2.cvtColor(cv2.imread(
            self.data_samples[idx].hdr_file, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
        ), cv2.COLOR_BGR2RGB)
        hdr_img = self.transform(hdr_img_raw)
        ldr_img = self.data_samples[idx].tmo(hdr_img)
        return hdr_img, ldr_img
    
    def transform(self, hdr_img: np.ndarray):
        return hdr_img


