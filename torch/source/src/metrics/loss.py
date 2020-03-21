import numpy as np

from .metric import Metric


class Loss(Metric):
    def __init__(self):
        super().__init__()
        self.init_val = np.inf

    def update(self, val_dict):
        loss = val_dict.loss.item()
        self.epoch_avg += loss
        self.running_avg += loss
        self.num_examples += val_dict.batch_size
        return loss
