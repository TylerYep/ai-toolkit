import torch
import torch.nn as nn
from .dice import DiceLoss
from .focal import FocalLoss


class MixedLoss(nn.Module):
    def __init__(self, alpha=8):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss()
        self.dice = DiceLoss()

    def forward(self, output, target):
        loss = self.alpha * self.focal(output, target) + self.dice(output, target)
        return loss.mean()
