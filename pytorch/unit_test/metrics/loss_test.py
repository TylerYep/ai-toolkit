""" accuracy_test.py """
from types import SimpleNamespace

from src.metrics import Loss


class TestLoss:
    @staticmethod
    def test_batch_loss(example_batch: SimpleNamespace) -> None:
        metric = Loss()

        _ = metric.update(example_batch)

        assert round(metric.get_batch_result(3), 3) == 0.21

    @staticmethod
    def test_epoch_accuracy(example_batch: SimpleNamespace) -> None:
        metric = Loss()

        for _ in range(4):
            example_batch.loss /= 2
            _ = metric.update(example_batch)

        assert str(metric) == "Loss: 0.0492"
