from .metric import Metric


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.epoch_acc = 0.0
        self.running_acc = 0.0
        self.num_examples = 0

    def formatted(self, computed_val):
        return f'{self.name}: {100. * computed_val:.2f}%'

    def reset(self):
        self.running_acc = 0.0

    def update(self, val_dict):
        output, target = val_dict['output'], val_dict['target']
        batch_size = val_dict['batch_size']
        accuracy = (output.argmax(1) == target).float().sum().item()
        self.epoch_acc += accuracy
        self.running_acc += accuracy
        self.num_examples += batch_size
        return accuracy

    def get_batch_result(self, log_interval, batch_size):
        return self.running_acc / (log_interval * batch_size)

    def get_epoch_result(self):
        return self.epoch_acc / self.num_examples
