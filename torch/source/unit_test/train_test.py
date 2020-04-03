''' train_test.py '''
from src.train import train

class TestTrain:

    @staticmethod
    def test_one_epoch():
        config = ['--no-visualize', '--num-examples=100']
        val_loss = train(['--epoch=1', '--name=TEST'] + config)
        assert round(val_loss, 7) == 4.4841685

    @staticmethod
    def test_epoch_resume():
        config = ['--no-visualize', '--num-examples=100']
        _ = train(['--epoch=2', '--name=TEST'] + config)
        val_loss_end = train(['--epoch=2', '--checkpoint=TEST'] + config)

        val_loss_test = train(['--epoch=4'] + config)

        assert val_loss_test == val_loss_end
