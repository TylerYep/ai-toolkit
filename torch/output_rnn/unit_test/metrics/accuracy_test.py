''' accuracy_test.py '''
import torch
from src.metrics import Accuracy


class TestAccuracy:
    @staticmethod
    def test_calculate_accuracy(example_output):
        output, target = example_output.output, example_output.target

        accuracy = Accuracy.calculate_accuracy(output, target)

        assert accuracy == 2.

    @staticmethod
    def test_batch_accuracy(example_output):
        metric = Accuracy()

        _ = metric.update(example_output)

        assert metric.get_batch_result(3) == 2/3

    @staticmethod
    def test_epoch_accuracy(example_output):
        metric = Accuracy()

        _ = metric.update(example_output)

        assert str(metric) == 'Accuracy: 66.67%'
