from types import SimpleNamespace

import torch

from .metric import Metric


class Accuracy(Metric):
    def __repr__(self) -> str:
        return f"{self.name}: {100. * self.value:.2f}%"

    @staticmethod
    def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
        return (output.argmax(1) == target).float().sum().item()

    def update(self, val_dict: SimpleNamespace) -> float:
        output, target = val_dict.output, val_dict.target
        accuracy = self.calculate_accuracy(output, target)
        self.epoch_avg += accuracy
        self.running_avg += accuracy
        self.num_examples += val_dict.batch_size
        return accuracy
