from .expandnet import ExpandNet, ExpandNetLoss


def model_factory(cfg):
    if cfg.model.name == "expandnet":
        return ExpandNet()
    else:
        raise NotImplementedError

def loss_fn_factory(cfg):
    if cfg.model.name == "expandnet":
        return ExpandNetLoss
    else:
        raise NotImplementedError

