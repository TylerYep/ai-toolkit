import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, output, target):
        max_val = (-output).clamp(min=0)
        loss = (
            output
            - output * target
            + max_val
            + ((-max_val).exp() + (-output - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-output * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()
