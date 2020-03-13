from .metric import Metric


class IoU(Metric):
    def __init__(self):
        super().__init__()
        self.epoch_acc = 0.0
        self.running_acc = 0.0

    @staticmethod
    def calculate_iou(output, target, eps=1e-7):
        output = output > 0.5
        output, target = output.squeeze(), target.squeeze()
        intersection = (output & (target).bool()).float().sum((1, 2)) + eps
        union = (output | target.bool()).float().sum((1, 2)) + eps
        accuracy = (intersection / union).sum().item()
        return accuracy

    def reset(self):
        self.running_acc = 0.0

    def update(self, val_dict):
        output, target = val_dict['output'], val_dict['target']
        accuracy = self.calculate_iou(output, target)
        self.epoch_acc += accuracy
        self.running_acc += accuracy
        return accuracy

    def get_batch_result(self, log_interval):
        return self.running_acc / log_interval

    def get_epoch_result(self, num_examples):
        return self.epoch_acc / num_examples
