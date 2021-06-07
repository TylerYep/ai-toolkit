from types import SimpleNamespace

import torch

from .metric import Metric


class Dice(Metric):
    @staticmethod
    def calculate_dice_coefficent(
        output: torch.Tensor, target: torch.Tensor, eps: float = 1e-7
    ) -> float:
        output = (output > 0.5).float()
        batch_size = output.shape[0]
        dice_target = target.reshape(batch_size, -1)
        dice_output = output.reshape(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1)
        accuracy = ((2 * intersection + eps) / (union + eps)).sum().item()
        return accuracy

    def update(self, val_dict: SimpleNamespace) -> float:
        output, target = val_dict.output, val_dict.target
        dice_score = self.calculate_dice_coefficent(output, target)
        self.epoch_avg += dice_score
        self.running_avg += dice_score
        self.num_examples += val_dict.batch_size
        return dice_score
