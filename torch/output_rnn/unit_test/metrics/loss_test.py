''' accuracy_test.py '''
import torch
from src.metrics import Loss


class TestLoss:
    @staticmethod
    def test_batch_loss(example_output):
        metric = Loss()

        _ = metric.update(example_output)

        assert round(metric.get_batch_result(3), 3) == 0.2

    @staticmethod
    def test_epoch_accuracy(example_output):
        metric = Loss()

        for _ in range(4):
            example_output.loss /= 2
            _ = metric.update(example_output)

        assert str(metric) == 'Loss: 0.0469'
