from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.transforms.v2 import Compose, Normalize, ToTensor

from . import BaseDataset, DataSample
from .augmentation import (apply_hue_jitter, random_crop, random_flip,
                           random_rotate)
from .utils import load_hdr


class PanoHdrDataset(BaseDataset):
    def __init__(self, data_samples: List[DataSample], **kwargs):
        super().__init__(**kwargs)
        self.crop = kwargs.get("crop", False)
        self.flip = kwargs.get("flip", False)
        self.rotate = kwargs.get("rotate", False)
        self.color_jitter = kwargs.get("color_jitter", False)
        self.img_size = kwargs.get("img_size", [1024, 512])
        self.data_samples = data_samples

        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        data_sample = self.data_samples[idx]
        hdr_img = np.log(load_hdr(data_sample.hdr_file) + 1e-6)
        ldr_img = np.array(Image.open(data_sample.ldr_file))
        hdr_img, ldr_img = self.augment(hdr_img, ldr_img)
        if hdr_img.shape[:2] != self.img_size[::-1]:
            hdr_img = cv2.resize(hdr_img, self.img_size)
            ldr_img = cv2.resize(ldr_img, self.img_size)
        return Tensor(hdr_img).permute(2, 0, 1), self.img_transform(ldr_img)
    
    def augment(self, hdr_img: np.ndarray, ldr_img: np.ndarray):
        if self.flip:
            hdr_img, ldr_img = random_flip(hdr_img, ldr_img)
        if self.rotate:
            hdr_img, ldr_img = random_rotate(hdr_img, ldr_img)
        if self.crop:
            hdr_img, ldr_img = random_crop(hdr_img, ldr_img, self.img_size[::-1])
        if self.color_jitter:
            hdr_img, ldr_img = apply_hue_jitter(hdr_img, ldr_img)
        return hdr_img, ldr_img
