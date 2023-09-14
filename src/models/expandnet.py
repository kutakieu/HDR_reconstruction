import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ExpandNet(nn.Module):
    def __init__(self, img_w: int=512, img_h: int=512, dim: int=64):
        super(ExpandNet, self).__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.dim = dim

        self.local_net = LocalBranch(dim=dim)
        self.mid_net = DilationBranch(dim=dim)
        self.global_net = GlobalBranch(img_w=img_w, img_h=img_h, dim=dim)
        self.end_net = Fusion()

    def forward(self, x):
        local = self.local_net(x)
        mid = self.mid_net(x)
        b, c, h, w = local.shape
        _global = self.global_net(x, b, h, w)
        fuse = torch.cat((local, mid, _global), -3)
        return self.end_net(fuse)
    

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, k, s, p, d=1):
        super(Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, k, s, p, d), 
            nn.SELU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class LocalBranch(nn.Module):
    def __init__(self, dim: int=64):
        super(LocalBranch, self).__init__()
        self.local_net = nn.Sequential(
            Block(3, dim, 3, 1, 1), 
            Block(dim, dim*2, 3, 1, 1)
        )

    def forward(self, x):
        return self.local_net(x)

class DilationBranch(nn.Module):
    def __init__(self, dim: int=64):
        super(DilationBranch, self).__init__()
        self.mid_net = nn.Sequential(
            Block(3, dim, 3, 1, 2, 2),
            Block(dim, dim, 3, 1, 2, 2),
            Block(dim, dim, 3, 1, 2, 2),
            nn.Conv2d(dim, dim, 3, 1, 2, 2),
        )
    def forward(self, x):
        return self.mid_net(x)

class GlobalBranch(nn.Module):
    def __init__(self, img_w: int=512, img_h: int=512, dim: int=64):
        self.img_w = img_w
        self.img_h = img_h
        self.dim = dim
        super(GlobalBranch, self).__init__()
        self.global_net = nn.Sequential(
            Block(3, dim, 3, 2, 1),
            Block(dim, dim, 3, 2, 1),
            Block(dim, dim, 3, 2, 1),
            Block(dim, dim, 3, 2, 1),
            Block(dim, dim, 3, 2, 1),
            Block(dim, dim, 3, 2, 1),
            Block(dim, dim, 3, 2, 1),
            nn.Conv2d(dim, dim, 4, 1, 0),
        )
    
    def forward(self, x, bs, h, w):
        resized = F.interpolate(
            x, (self.img_h, self.img_w), mode='bilinear', align_corners=False
        )
        return self.global_net(resized).expand(bs, self.dim, h, w)

class Fusion(nn.Module):
    def __init__(self, dim: int=64):
        super(Fusion, self).__init__()
        self.end_net = nn.Sequential(
            Block(dim*4, dim, 1, 1, 0), 
            nn.Conv2d(dim, 3, 1, 1, 0), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.end_net(x)

class ExpandNetLoss(nn.Module):
    def __init__(self, loss_lambda=5):
        super(ExpandNetLoss, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l1_loss = nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, pred, gt):
        cosine_term = (1 - self.similarity(pred, gt)).mean()
        return self.l1_loss(pred, gt) + self.loss_lambda * cosine_term
