''' epoch_test.py '''
from src.train import train

class TestRNN:

    @staticmethod
    def test_one_epoch():
        val_loss = train(['--epoch=1', '--name=TEST'])
        assert round(val_loss, 7) == 0.0134822

    @staticmethod
    def test_epoch_resume():
        _ = train(['--epoch=2', '--name=TEST'])
        val_loss_end = train(['--epoch=2', '--checkpoint=TEST'])

        val_loss_test = train(['--epoch=4'])

        assert val_loss_test == val_loss_end
