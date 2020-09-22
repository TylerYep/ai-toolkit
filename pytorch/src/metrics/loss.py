from types import SimpleNamespace
from typing import cast

from .metric import Metric


class Loss(Metric):
    def update(self, val_dict: SimpleNamespace) -> float:
        loss = val_dict.loss.item()
        self.epoch_avg += loss * val_dict.batch_size
        self.running_avg += loss * val_dict.batch_size
        self.num_examples += val_dict.batch_size
        return cast(float, loss)
