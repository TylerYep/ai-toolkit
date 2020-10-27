""" train_test.py """
from src.train import train


class TestTrain:
    @staticmethod
    def test_one_epoch() -> None:
        config = ["--no-visualize", "--num-examples=100", "--no-save"]

        metric_tracker = train(["--epoch=1", "--name=TEST"] + config)

        assert round(metric_tracker["Loss"].value, 7) == 4.4703889

    @staticmethod
    def test_epoch_resume(tmp_path: str) -> None:
        config = ["--no-visualize", "--num-examples=100", f"--save-dir={tmp_path}"]
        _ = train(["--epoch=2", "--name=TEST"] + config)
        metrics_end = train(["--epoch=2", "--checkpoint=TEST"] + config)

        metrics_test = train(["--epoch=4"] + config)

        assert metrics_test == metrics_end
