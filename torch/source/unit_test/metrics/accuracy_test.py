''' accuracy_test.py '''
from src.metrics import Accuracy


class TestAccuracy:
    @staticmethod
    def test_calculate_accuracy(example_batch):
        output, target = example_batch.output, example_batch.target

        accuracy = Accuracy.calculate_accuracy(output, target)

        assert accuracy == 2.

    @staticmethod
    def test_batch_accuracy(example_batch):
        metric = Accuracy()

        _ = metric.update(example_batch)

        assert metric.get_batch_result(3) == 2/3

    @staticmethod
    def test_epoch_accuracy(example_batch):
        metric = Accuracy()

        _ = metric.update(example_batch)

        assert str(metric) == 'Accuracy: 66.67%'
