from random import random, uniform
from typing import Optional, Tuple

import cv2
import numpy as np

from .e2p import e2p_with_map, generate_e2p_map


def random_scale(img: np.ndarray, img2: Optional[np.ndarray]):
    scale = uniform(0.5, 2)
    h, w = img.shape[:2]
    if img2 is None:
        return cv2.resize(img, (int(w*scale), int(h*scale)))
    h2, w2 = img2.shape[:2]
    return cv2.resize(img, (int(w*scale), int(h*scale))), cv2.resize(img2, (int(w2*scale), int(h2*scale)))


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
    

def random_flip(img: np.ndarray, img2: Optional[np.ndarray], force=False):
    """Flip panorama image horizontally
    Args:
        img (np.ndarray): input image
    Returns:
        np.ndarray: horizontally flipped image
    """
    if random() < 0.5 or force:
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

def apply_hue_jitter(img: np.ndarray, img2: Optional[np.ndarray], hue_jitter: float=0.05):
    """Apply color jitter to image
    Args:
        img (np.ndarray): input image
        img2 Optional(np.ndarray): input image
    Returns:
        np.ndarray: color jittered image
    """
    img = img.copy()
    hue_jitter = np.random.randint(-360*hue_jitter, 360*hue_jitter)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:,:,0] = (img[:,:,0] + hue_jitter) % 360.0
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    if img2 is None:
        return img
    img2 = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    img2[:,:,0] = (img2[:,:,0] + hue_jitter) % 360.0
    img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2RGB)
    return img, img2

def random_e2p(hdr_img, ldr_img, force=False):
    if random() < 0.5 or force:
        e2p_map = generate_e2p_map(
            in_hw=hdr_img.shape[:2],
            fov_deg=np.random.randint(60, 90),
            u_deg=0,
            v_deg=np.random.randint(-45, 45),
            out_hw=[hdr_img.shape[1]//4, hdr_img.shape[1]//4],
        )
        return e2p_with_map(hdr_img, e2p_map), e2p_with_map(ldr_img, e2p_map)
    return hdr_img, ldr_img
