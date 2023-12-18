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
    if loss_function_name == "scale_invariant_mse":
        return scale_invariant_mse_loss

def scale_invariant_mse_loss(pred, gt, eps=1e-6):
    """Scale invariant MSE loss
    Args:
        pred (torch.Tensor): predicted image
        gt (torch.Tensor): ground truth image
        eps (float, optional): epsilon value. Defaults to 1e-6.
    Returns:
        torch.Tensor: scale invariant MSE loss
    """
    return torch.mean(
        torch.pow(torch.log(pred + eps) - torch.log(gt + eps), 2)
        - torch.mean(torch.log(pred + eps) - torch.log(gt + eps)) ** 2
    )
