from abc import ABC

import cv2
import numpy as np
from numpy.random import uniform


class BaseToneMapper(ABC):
    def __call__(self, img):
        return self.map_range(self.tonemapper.process(img))
    
    @staticmethod
    def map_range(x: np.ndarray, low: float=0, high: float=1):
        return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)

class Exposure(BaseToneMapper):
    def __init__(self, 
                 stops: float=0.0, 
                 gamma: float=1.0, 
                 randomize: bool=False
                 ):
        if randomize:
            gamma = uniform(1.0, 1.2)
        self.stops = stops
        self.gamma = gamma

    def __call__(self, img):
        return np.clip(img * (2 ** self.stops), 0, 1) ** self.gamma


class PercentileExposure(BaseToneMapper):
    def __init__(self, 
                 gamma: float=2.0, 
                 low_perc: float=10, 
                 high_perc: float=90, 
                 randomize: bool=False
                 ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            low_perc = uniform(0, 15)
            high_perc = uniform(85, 100)
        self.gamma = gamma
        self.low_perc = low_perc
        self.high_perc = high_perc

    def __call__(self, img):
        low, high = np.percentile(img, (self.low_perc, self.high_perc))
        return self.map_range(np.clip(img, low, high)) ** (1 / self.gamma)


class Reinhard(BaseToneMapper):
    def __init__(self,
                 intensity: float=-1.0,
                 light_adapt: float=0.8,
                 color_adapt: float=0.0,
                 gamma: float=2.0,
                 randomize: bool=False,
                 ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            intensity = uniform(-1.0, 1.0)
            light_adapt = uniform(0.8, 1.0)
            color_adapt = uniform(0.0, 0.2)
        self.tonemapper = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt,
        )


class Mantiuk(BaseToneMapper):
    def __init__(self, 
                 saturation: float=1.0, 
                 scale: float=0.75, 
                 gamma: float=2.0, 
                 randomize: bool=False
                 ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            scale = uniform(0.65, 0.85)

        self.tonemapper = cv2.createTonemapMantiuk(
            saturation=saturation, scale=scale, gamma=gamma
        )


class Drago(BaseToneMapper):
    def __init__(self, 
                 saturation: float=1.0, 
                 bias: float=0.85, 
                 gamma: float=2.0, 
                 randomize: bool=False
                 ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            bias = uniform(0.7, 0.9)

        self.tonemapper = cv2.createTonemapDrago(
            saturation=saturation, bias=bias, gamma=gamma
        )


class Durand(BaseToneMapper):
    def __init__(
        self,
        gamma: float=2.,
        randomize: bool=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
        self.tonemapper = cv2.createTonemap(
            gamma=gamma,
        )


TM_DICT = {
    "Exposure": Exposure,
    "PercentileExposure": PercentileExposure,
    "Reinhard": Reinhard,
    "Mantiuk": Mantiuk,
    "Drago": Drago,
    "Durand": Durand,
}


if __name__ == "__main__":
    from PIL import Image

    hdr_file = "tests/data/brown_photostudio_02_0.5k.hdr"

    hdr_img = cv2.cvtColor(
        cv2.imread(str(hdr_file), flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR), 
        cv2.COLOR_BGR2RGB
        )
    # ldr_img = Reinhard(randomize=False)(hdr_img)
    ldr_img = Reinhard(
        intensity=-1.0,
        light_adapt=0.8,
        color_adapt=0.,
        gamma=2.,
        randomize=False,
    )(hdr_img)
    
    # for tm_name, tm in TM_DICT.items():
    #     ldr_img = tm(randomize=False)(hdr_img)
    #     Image.fromarray((255 * ldr_img).astype(np.uint8)).save(f"ldr_{tm_name}.png")
