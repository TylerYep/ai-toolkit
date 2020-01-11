from torch.nn import functional as F

from .metric import Metric

class F1Score(Metric):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.epsilon = eps
        self.epoch_f1 = 0.0
        self.running_f1 = 0.0

    def reset(self):
        self.running_f1 = 0.0

    def update(self, val_dict):
        y_pred, y_true = val_dict['output'], val_dict['target']

        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0)
        # tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        f1_score = 1 - f1.mean()

        self.epoch_f1 += f1_score
        self.running_f1 += f1_score
        return f1_score

    def get_batch_result(self, log_interval):
        return self.running_f1 / log_interval

    def get_epoch_result(self, num_examples):
        return self.epoch_f1 / num_examples
