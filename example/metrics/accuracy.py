from .metric import Metric

class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.epoch_acc = 0.0
        self.running_acc = 0.0

    def __repr__(self):
        return self.running_acc

    def formatted(self, computed_val):
        return f'{self.name}: {computed_val:.2f}%'

    def reset(self):
        self.running_acc = 0.0

    def update(self, val_dict):
        self.epoch_acc += val_dict[self.name]
        self.running_acc += val_dict[self.name]

    def get_batch_result(self, log_interval):
        return self.running_acc / log_interval

    def get_epoch_result(self, num_examples):
        return 100. * self.epoch_acc / num_examples
