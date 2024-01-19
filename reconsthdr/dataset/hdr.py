from typing import List, Tuple

import cv2
import numpy as np
from torch import Tensor

from ..utils import load_hdr
from . import BaseDataset, DataSample
from .augmentation import (apply_hue_jitter, random_crop, random_e2p,
                           random_flip, random_rotate)


def convert_ldr_to_tensor(ldr_img: np.ndarray):
    return Tensor(ldr_img * 2 - 1.0).permute(2, 0, 1)

class PanoHdrDataset(BaseDataset):
    def __init__(self, data_samples: List[DataSample], **kwargs):
        super().__init__(**kwargs)
        self.crop = kwargs.get("crop", False)
        self.flip = kwargs.get("flip", False)
        self.rotate = kwargs.get("rotate", False)
        self.color_jitter = kwargs.get("color_jitter", False)
        self.e2p = kwargs.get("e2p", False)
        self.img_size = kwargs.get("img_size", [1024, 512])
        self.data_samples = data_samples
    
    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        data_sample = self.data_samples[idx]
        hdr_img = np.log(load_hdr(data_sample.hdr_file) + 1e-6)
        ldr_img = load_hdr(data_sample.ldr_file)
        hdr_img, ldr_img = self.augment(hdr_img, ldr_img)
        if hdr_img.shape[:2] != self.img_size[::-1]:
            hdr_img = cv2.resize(hdr_img, self.img_size)
            ldr_img = cv2.resize(ldr_img, self.img_size)
        return Tensor(hdr_img).permute(2, 0, 1), convert_ldr_to_tensor(ldr_img)
    
    def augment(self, hdr_img: np.ndarray, ldr_img: np.ndarray):
        if self.flip:
            hdr_img, ldr_img = random_flip(hdr_img, ldr_img)
        if self.rotate:
            hdr_img, ldr_img = random_rotate(hdr_img, ldr_img)
        if self.e2p:
            hdr_img, ldr_img = random_e2p(hdr_img, ldr_img)
        if self.crop:
            hdr_img, ldr_img = random_crop(hdr_img, ldr_img, self.img_size[::-1])
        if self.color_jitter:
            hdr_img, ldr_img = apply_hue_jitter(hdr_img, ldr_img)
        return hdr_img, ldr_img
