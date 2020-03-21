from .metric import Metric


class Accuracy(Metric):
    def __init__(self):
        super().__init__()

    @staticmethod
    def calculate_accuracy(output, target):
        return (output.argmax(1) == target).float().sum().item()

    def formatted(self, computed_val):
        return f'{self.name}: {100. * computed_val:.2f}%'

    def update(self, val_dict):
        output, target = val_dict.output, val_dict.target
        accuracy = self.calculate_accuracy(output, target)
        self.epoch_avg += accuracy
        self.running_avg += accuracy
        self.num_examples += val_dict.batch_size
        return accuracy
