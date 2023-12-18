"""original code: https://github.com/dmarnerides/hdr-expandnet"""

import cv2
import numpy as np
from numpy.random import uniform


def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)


class Exposure(object):
    def __init__(self, stops=0.0, gamma=1.0):
        self.stops = stops
        self.gamma = gamma

    def process(self, img):
        return np.clip(img * (2 ** self.stops), 0, 1) ** self.gamma


class PercentileExposure(object):
    def __init__(self, gamma=2.0, low_perc=10, high_perc=90, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            low_perc = uniform(0, 15)
            high_perc = uniform(85, 100)
        self.gamma = gamma
        self.low_perc = low_perc
        self.high_perc = high_perc

    def __call__(self, x):
        low, high = np.percentile(x, (self.low_perc, self.high_perc))
        return map_range(np.clip(x, low, high)) ** (1 / self.gamma)


class BaseTMO(object):
    def __call__(self, img):
        return map_range(self.op.process(img))


class Reinhard(BaseTMO):
    def __init__(
        self,
        intensity=-1.0,
        light_adapt=0.8,
        color_adapt=0.0,
        gamma=2.0,
        randomize=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            intensity = uniform(-1.0, 1.0)
            light_adapt = uniform(0.8, 1.0)
            color_adapt = uniform(0.0, 0.2)
        self.op = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt,
        )


class Mantiuk(BaseTMO):
    def __init__(self, saturation=1.0, scale=0.75, gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            scale = uniform(0.65, 0.85)

        self.op = cv2.createTonemapMantiuk(
            saturation=saturation, scale=scale, gamma=gamma
        )


class Drago(BaseTMO):
    def __init__(self, saturation=1.0, bias=0.85, gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            bias = uniform(0.7, 0.9)

        self.op = cv2.createTonemapDrago(
            saturation=saturation, bias=bias, gamma=gamma
        )


class Durand(BaseTMO):
    def __init__(
        self,
        contrast=3,
        saturation=1.0,
        sigma_space=8,
        sigma_color=0.4,
        gamma=2.0,
        randomize=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            contrast = uniform(3.5)
        self.op = cv2.createTonemapDurand(
            contrast=contrast,
            saturation=saturation,
            sigma_space=sigma_space,
            sigma_color=sigma_color,
            gamma=gamma,
        )
