from .expandnet import ExpandNet
from .unet import Unet


def model_factory(cfg):
    if cfg.model.name == "expandnet":
        return ExpandNet()
    elif cfg.model.name == "unet":
        return Unet()
    else:
        raise NotImplementedError

