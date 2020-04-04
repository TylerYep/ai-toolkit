import numpy as np

from .metric import Metric


class Loss(Metric):
    def __init__(self):
        super().__init__()
        self.value = np.inf

    def update(self, val_dict):
        loss = val_dict.loss.item()
        self.epoch_avg += loss * val_dict.batch_size
        self.running_avg += loss * val_dict.batch_size
        self.num_examples += val_dict.batch_size
        return loss
