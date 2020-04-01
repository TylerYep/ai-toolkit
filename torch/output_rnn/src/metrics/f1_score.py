from torch.nn import functional as F

from .metric import Metric


class F1Score(Metric):
    @staticmethod
    def calculate_f1_score(y_pred, y_true, eps=1e-7):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0)
        # tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        f1 = 2 * (precision * recall) / (precision + recall + eps)
        f1 = f1.clamp(min=eps, max=1-eps)
        f1_score = 1 - f1.mean()
        return f1_score

    def update(self, val_dict):
        y_pred, y_true = val_dict.output, val_dict.target
        f1_score = self.calculate_f1_score(y_pred, y_true)
        self.epoch_avg += f1_score
        self.running_avg += f1_score
        self.num_examples += val_dict.batch_size
        return f1_score
