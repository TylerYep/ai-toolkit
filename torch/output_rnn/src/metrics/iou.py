from .metric import Metric


class IoU(Metric):
    @staticmethod
    def calculate_iou(output, target, eps=1e-7):
        output = output > 0.5
        output, target = output.squeeze(), target.squeeze().bool()
        intersection = (output & target).float().sum((1, 2)) + eps
        union = (output | target).float().sum((1, 2)) + eps
        accuracy = (intersection / union).sum().item()
        return accuracy

    def update(self, val_dict):
        output, target = val_dict.output, val_dict.target
        accuracy = self.calculate_iou(output, target)
        self.epoch_avg += accuracy
        self.running_avg += accuracy
        self.num_examples += val_dict.batch_size
        return accuracy
