
import numpy as np

from reconsthdr.dataset.augmentation import (random_crop, random_e2p,
                                             random_flip, random_rotate,
                                             random_scale)


def test_random_scale():
    img = np.random.rand(512, 1024, 3)
    img2 = img.copy()
    scaled_img, scaled_img2 = random_scale(img, img2)
    np.testing.assert_array_equal(scaled_img, scaled_img2)

def test_random_crop():
    img = np.random.rand(512, 1024, 3)
    img2 = img.copy()
    crop_hw = (256, 256)
    cropped_img, cropped_img2 = random_crop(img, img2, crop_hw)
    assert cropped_img.shape == (256, 256, 3)
    assert cropped_img2.shape == (256, 256, 3)
    np.testing.assert_array_equal(cropped_img, cropped_img2)

def test_random_flip():
    img = np.random.rand(512, 1024, 3)
    img2 = img.copy()
    flipped_img, flipped_img2 = random_flip(img, img2, force=True)
    assert flipped_img.shape == (512, 1024, 3)
    assert flipped_img2.shape == (512, 1024, 3)
    np.testing.assert_array_equal(flipped_img, flipped_img2)

def test_random_rotate():
    img = np.random.rand(512, 1024, 3)
    img2 = img.copy()
    rotated_img, rotated_img2 = random_rotate(img, img2)
    assert rotated_img.shape == (512, 1024, 3)
    assert rotated_img2.shape == (512, 1024, 3)
    np.testing.assert_array_equal(rotated_img, rotated_img2)

def test_random_e2p():
    img = np.random.rand(512, 1024, 3)
    img2 = img.copy()
    e2p_img, e2p_img2 = random_e2p(img, img2, force=True)
    np.testing.assert_array_equal(e2p_img, e2p_img2)
