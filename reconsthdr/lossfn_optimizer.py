from typing import Literal

import torch
import torch.optim as optim

OptimizerType = Literal["sgd", "adam"]


def optimizer_factory(optimizer_name: OptimizerType, learning_rate: float, net):
    if optimizer_name == "sgd":
        return optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "adam":
        return optim.Adam(net.parameters(), lr=learning_rate)

def loss_function_factory(loss_function_name: str):
    if loss_function_name == "simse":
        return scale_invariant_mse_loss
    elif loss_function_name == "simse+mae":
        return simse_and_mae_loss
    elif loss_function_name == "mse":
        return mse_loss

def scale_invariant_mse_loss(pred, gt):
    """Scale invariant MSE loss
    Args:
        pred (torch.Tensor): predicted image
        gt (torch.Tensor): ground truth image
        eps (float, optional): epsilon value. Defaults to 1e-6.
    Returns:
        torch.Tensor: scale invariant MSE loss
    """
    return torch.mean((pred - gt) ** 2) - torch.mean(pred - gt) ** 2

def simse_and_mae_loss(pred, gt):
    simse_loss = scale_invariant_mse_loss(pred, gt)
    mae_loss = torch.mean(torch.abs(pred - gt))
    return simse_loss + mae_loss

def mse_loss(pred, gt):
    return torch.mean((pred - gt) ** 2)
