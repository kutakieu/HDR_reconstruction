from random import random
from typing import Optional, Tuple

import numpy as np
from torch import Tensor
from torchvision.transforms.v2 import ColorJitter


def random_crop(img: np.ndarray, img2: Optional[np.ndarray], crop_hw: Tuple[int, int]):
    """Randomly crop panorama image
    Args:
        img (np.ndarray): input image
    Returns:
        np.ndarray: cropped image
    """
    crop_h, crop_w = crop_hw
    h, w, _ = img.shape
    x = np.random.randint(0, w - crop_w)
    y = np.random.randint(0, h - crop_h)
    if img2 is None:
        return img[y:y+crop_h, x:x+crop_w, :]
    return img[y:y+crop_h, x:x+crop_w, :], img2[y:y+crop_h, x:x+crop_w, :]
    

def random_flip(img: np.ndarray, img2: Optional[np.ndarray]):
    """Flip panorama image horizontally
    Args:
        img (np.ndarray): input image
    Returns:
        np.ndarray: horizontally flipped image
    """
    if random() < 0.5:
        if img2 is None:
            return np.flip(img, axis=1)
        return np.flip(img, axis=1), np.flip(img2, axis=1)
    if img2 is None:
        return img
    return img, img2
    

def random_rotate(img: np.ndarray, img2: Optional[np.ndarray]) -> np.ndarray:
    """Randomly rotate image horizontally
    Args:
        img (np.ndarray): input image
    Returns:
        np.ndarray: horizontally rotated panorama image
    """
    rotate_pixels = np.random.randint(img.shape[1])
    if img2 is None:
        return np.roll(img, rotate_pixels, axis=1)
    return np.roll(img, rotate_pixels, axis=1), np.roll(img2, rotate_pixels, axis=1)

def apply_hue_jitter(img: np.ndarray, img2: Optional[np.ndarray], color_jitter: Optional[ColorJitter]=None):
    """Apply color jitter to image
    Args:
        img (np.ndarray): input image
        color_jitter (ColorJitter): color jitter transform
    Returns:
        np.ndarray: color jittered image
    """
    color_jitter = color_jitter or ColorJitter(hue=0.1)
    if img2 is None:
        return color_jitter(Tensor(img, device="cpu").permute(2, 0, 1)).permute(1, 2, 0).numpy()
    return color_jitter(Tensor(img, device="cpu").permute(2, 0, 1)).permute(1, 2, 0).numpy(), \
        color_jitter(Tensor(img2, device="cpu").permute(2, 0, 1)).permute(1, 2, 0).numpy()
