import numpy as np


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
