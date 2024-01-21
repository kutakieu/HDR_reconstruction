from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_bgr_ldr(img_file: Union[str, Path]):
    """load BGR format LDR image as RGB format numpy array"""
    return cv2.imread(str(img_file)), 

def load_rgb_ldr(img_file: Union[str, Path]):
    """load RGB format LDR image as RGB format numpy array"""
    return cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)

def load_bgr_hdr(img_file: Union[str, Path]):
    """load BGR format HDR image as RGB format numpy array"""
    return cv2.imread(str(img_file), flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)

def load_rgb_hdr(img_file: Union[str, Path]):
    """load RGB format HDR image as RGB format numpy array"""
    return cv2.cvtColor(
        cv2.imread(str(img_file), flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR), 
        cv2.COLOR_BGR2RGB
        )

def save_hdr(filepath: Union[str, Path], hdr_img: np.ndarray):
    """save RGB format HDR image as RGB format image"""
    cv2.imwrite(
        str(filepath), 
        cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
        )

def save_ldr(filepath: Union[str, Path], hdr_img: np.ndarray):
    """save RGB format LDR image as RGB format image"""
    cv2.imwrite(
        str(filepath), 
        cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
        )

def match_image_size(img1: np.ndarray, img2: np.ndarray):
    if img1.shape[:2] != img2.shape[:2]:
        smaller_shape = np.min([img1.shape[:2], img2.shape[:2]], axis=0)
        hdr_resized = cv2.resize(img1, (smaller_shape[1], smaller_shape[0]))
        ldr_resized = cv2.resize(img2, (smaller_shape[1], smaller_shape[0]))
        return hdr_resized, ldr_resized
    return img1, img2
