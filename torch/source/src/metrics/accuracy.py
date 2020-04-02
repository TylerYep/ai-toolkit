from .metric import Metric


class Accuracy(Metric):
    @staticmethod
    def calculate_accuracy(output, target):
        return (output.argmax(1) == target).float().sum().item()

    def __repr__(self):
        return f'{self.name}: {100. * self.get_epoch_result():.2f}%'

    def update(self, val_dict):
        output, target = val_dict.output, val_dict.target
        accuracy = self.calculate_accuracy(output, target)
        self.epoch_avg += accuracy
        self.running_avg += accuracy
        self.num_examples += val_dict.batch_size
        return accuracy
