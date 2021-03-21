""" train_test.py """
from pathlib import Path

from src.train import train


class TestTrain:
    @staticmethod
    def test_one_epoch() -> None:
        metric_tracker = train(
            "--no-visualize",
            "--num-examples=100",
            "--no-save",
            "--epoch=1",
            "--checkpoint=TEST",
        )

        assert round(metric_tracker["Loss"].value, 7) == 4.4703889

    @staticmethod
    def test_epoch_resume(tmp_path: Path) -> None:
        config = ["--no-visualize", "--num-examples=100", f"--save-dir={tmp_path}"]
        _ = train("--epoch=2", "--checkpoint=TEST", *config)
        metrics_end = train("--epoch=2", "--checkpoint=TEST", *config)

        metrics_test = train("--epoch=4", *config)

        assert metrics_test == metrics_end

    @staticmethod
    def test_configs(tmp_path: Path) -> None:
        metric_tracker = train("--config=test", f"--save-dir={tmp_path}")

        assert Path(metric_tracker.run_name).name == "A"
