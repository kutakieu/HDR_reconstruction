from typing import Optional

import numpy as np
from torch import Tensor
from torchvision.transforms.v2 import ColorJitter


def flip(img: np.ndarray):
    """Flip panorama image horizontally
    Args:
        img (np.ndarray): input image
    Returns:
        np.ndarray: horizontally flipped image
    """
    return np.flip(img, axis=1)
    

def random_rotate(img: np.ndarray) -> np.ndarray:
    """Randomly rotate image horizontally
    Args:
        img (np.ndarray): input image
    Returns:
        np.ndarray: horizontally rotated panorama image
    """
    rotate_pixels = np.random.randint(img.shape[1])
    return np.roll(img, rotate_pixels, axis=1)

def apply_hue_jitter(img: np.ndarray, color_jitter: Optional[ColorJitter]=None):
    """Apply color jitter to image
    Args:
        img (np.ndarray): input image
        color_jitter (ColorJitter): color jitter transform
    Returns:
        np.ndarray: color jittered image
    """
    color_jitter = color_jitter or ColorJitter(hue=0.1)
    return color_jitter(Tensor(img, device="cpu").permute(2, 0, 1)).permute(1, 2, 0).numpy()
