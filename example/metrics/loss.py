from .metric import Metric

class Loss(Metric):
    def __init__(self):
        super().__init__()
        self.epoch_loss = 0.0
        self.running_loss = 0.0

    def formatted(self, computed_val):
        return f'{self.name}: {computed_val:.4f}'

    def reset(self):
        self.running_loss = 0.0

    def update(self, val_dict):
        self.epoch_loss += val_dict[self.name]
        self.running_loss += val_dict[self.name]

    def get_batch_result(self, log_interval):
        return self.running_loss / log_interval

    def get_epoch_result(self, num_examples):
        return self.epoch_loss / num_examples
