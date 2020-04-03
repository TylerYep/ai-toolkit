''' accuracy_test.py '''
from types import SimpleNamespace
import torch
from src.metrics import Accuracy


class TestAccuracy:
    @staticmethod
    def test_calculate_accuracy():
        output = torch.tensor([
            [1., 0, 0, 0, 0],
            [0.1, 0.7, 0.2, 0, 0],
            [0.1, 0.7, 0.8, 0, 0],
        ])
        target = torch.tensor([0, 1, 2])

        accuracy = Accuracy.calculate_accuracy(output, target)

        assert accuracy == 3.

    @staticmethod
    def test_batch_accuracy():
        metric = Accuracy()
        val_dict = SimpleNamespace(**{
            'output': torch.tensor([
                [1., 0, 0, 0, 0],
                [0.1, 0.7, 0.2, 0, 0],
                [0.1, 0.7, 0, 0, 0.8],
            ]),
            'target': torch.tensor([0, 1, 1]),
            'batch_size': 3
        })

        _ = metric.update(val_dict)

        assert metric.get_batch_result(3) == 2/3

    @staticmethod
    def test_epoch_accuracy():
        metric = Accuracy()
        val_dict = SimpleNamespace(**{
            'output': torch.tensor([
                [1., 0, 0, 0, 0],
                [0.1, 0.7, 0.2, 0, 0],
                [0.1, 0.7, 0.8, 0, 0],
            ]),
            'target': torch.tensor([0, 1, 1]),
            'batch_size': 3
        })

        _ = metric.update(val_dict)

        assert str(metric) == 'Accuracy: 66.67%'
