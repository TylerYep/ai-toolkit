from .metric import Metric
import torch

SMOOTH = 1e-7

class Dice(Metric):
    def __init__(self):
        super().__init__()
        self.epoch_acc = 0.0
        self.running_acc = 0.0

    def reset(self):
        self.running_acc = 0.0

    def update(self, val_dict):
        output, target = val_dict['output'], val_dict['target']
        output = output > 0.5
        batch_size = output.shape[0]
        dice_target = target.reshape(batch_size, -1)
        dice_output = output.reshape(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1)
        accuracy = ((2 * intersection + SMOOTH) / (union + intersection + SMOOTH)).sum().item()
        self.epoch_acc += accuracy
        self.running_acc += accuracy
        return accuracy

    def get_batch_result(self, log_interval):
        return self.running_acc / log_interval

    def get_epoch_result(self, num_examples):
        return self.epoch_acc / num_examples
