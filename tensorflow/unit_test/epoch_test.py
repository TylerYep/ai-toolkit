""" epoch_test.py """
from src.train import train


class TestCNN:
    @staticmethod
    def test_one_epoch() -> None:
        val_loss = train(["--epoch=1", "--name=TEST"])
        assert round(val_loss, 7) == 1.4522075

    @staticmethod
    def test_epoch_resume() -> None:
        _ = train(["--epoch=2", "--name=TEST"])
        val_loss_end = train(["--epoch=2", "--checkpoint=TEST"])

        val_loss_test = train(["--epoch=4"])

        assert val_loss_test == val_loss_end
