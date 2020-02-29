import numpy as np

from .metric import Metric


class Loss(Metric):
    def __init__(self):
        super().__init__()
        self.init_val = np.inf
        self.epoch_loss = 0.0
        self.running_loss = 0.0
        self.num_examples = 0

    def reset(self):
        self.running_loss = 0.0

    def update(self, val_dict):
        loss = val_dict['loss'].item()
        batch_size = val_dict['batch_size']
        self.epoch_loss += loss
        self.running_loss += loss
        self.num_examples += batch_size
        return loss

    def get_batch_result(self, log_interval, batch_size):
        return self.running_loss / (log_interval * batch_size)

    def get_epoch_result(self):
        return self.epoch_loss / self.num_examples
