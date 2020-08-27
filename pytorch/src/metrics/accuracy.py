from types import SimpleNamespace

import torch

from .metric import Metric


class Accuracy(Metric):
    @staticmethod
    def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
        accuracy = (output.argmax(1) == target).float().sum().item()
        assert isinstance(accuracy, float)
        return accuracy

    def __repr__(self) -> str:
        return f"{self.name}: {100. * self.value:.2f}%"

    def update(self, val_dict: SimpleNamespace) -> float:
        output, target = val_dict.output, val_dict.target
        accuracy = self.calculate_accuracy(output, target)
        self.epoch_avg += accuracy
        self.running_avg += accuracy
        self.num_examples += val_dict.batch_size
        return accuracy
