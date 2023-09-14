from typing import Literal

import torch.nn as nn
import torch.optim as optim

LossFunctionType = Literal["cross_entropy"]
OptimizerType = Literal["sgd", "adam"]


def loss_function_factory(loss_function_name: LossFunctionType):
    if loss_function_name == "cross_entropy":
        return nn.CrossEntropyLoss()


def optimizer_factory(optimizer_name: OptimizerType, learning_rate: float, net):
    if optimizer_name == "sgd":
        return optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "adam":
        return optim.Adam(net.parameters(), lr=learning_rate)
