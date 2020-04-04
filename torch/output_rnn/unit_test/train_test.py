''' train_test.py '''
from src.train import train

class TestTrain:

    @staticmethod
    def test_one_epoch():
        config = ['--no-visualize', '--num-examples=100']

        metrics = train(['--epoch=1', '--name=TEST'] + config)

        assert round(metrics.get_primary_metric(), 7) == 4.5108116

    @staticmethod
    def test_epoch_resume():
        config = ['--no-visualize', '--num-examples=100']
        _ = train(['--epoch=2', '--name=TEST'] + config)
        metrics_end = train(['--epoch=2', '--checkpoint=TEST'] + config)

        metrics_test = train(['--epoch=4'] + config)

        assert metrics_test.get_primary_metric() == metrics_end.get_primary_metric()
        # TODO override equality operator
