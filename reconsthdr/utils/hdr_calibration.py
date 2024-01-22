import numpy as np

from .image import match_image_size


def calibrate_hdr(hdr: np.ndarray, ldr: np.ndarray):
    """Calibrate HDR image using LDR image
    Args:
        hdr (np.ndarray): HDR image
        ldr (np.ndarray): LDR image
    Returns:
        np.ndarray: calibrated HDR image
    """
    if np.max(ldr) > 1:
        ldr = ldr / 255.0

    linear_rgb_ldr = sRGB2linearRGB(ldr)
    return calibrate_hdr_with_linear_rgb(hdr, linear_rgb_ldr)

def calibrate_hdr_with_linear_rgb(hdr: np.ndarray, linear_rgb_ldr: np.ndarray):
    hdr_resized, ldr_resized = match_image_size(hdr, linear_rgb_ldr)
    non_overexposed_mask = _get_non_overexposed_mask(ldr_resized)
    sum_ldr_non_overexposed = np.sum(ldr_resized[non_overexposed_mask, :])
    sum_hdr_non_overexposed = np.sum(hdr_resized[non_overexposed_mask, :])
    if sum_ldr_non_overexposed == 0 or sum_hdr_non_overexposed == 0:
        raise ValueError("HDR calibration failed")
    return hdr * (sum_ldr_non_overexposed / sum_hdr_non_overexposed)

def _get_non_overexposed_mask(ldr: np.ndarray, tau: float=0.83):
    return np.sum(ldr, axis=2) < tau*3

def sRGB2linearRGB(img: np.ndarray):
    if np.max(img) > 1:
        img = img / 255.0
    return np.where(img <= 0.04045, img / 12.92, np.power((img+0.055) / 1.055, 2.4))

def linearRGB2sRGB(img: np.ndarray):
    img = np.where(img <= 0.0031308, img * 12.92,
                   np.power(img, 1/2.4) * 1.055 - 0.055)
    return img * 255
