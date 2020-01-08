import numpy as np

from .metric import Metric

class Loss(Metric):
    def __init__(self):
        super().__init__()
        self.init_val = np.inf
        self.epoch_loss = 0.0
        self.running_loss = 0.0

    def reset(self):
        self.running_loss = 0.0

    def update(self, val_dict):
        loss = val_dict['loss'].item()
        self.epoch_loss += loss
        self.running_loss += loss
        return loss

    def get_batch_result(self, log_interval):
        return self.running_loss / log_interval

    def get_epoch_result(self, num_examples):
        return self.epoch_loss / num_examples
