""" train_test.py """
from types import SimpleNamespace

from _pytest.capture import CaptureFixture

from src.args import init_pipeline
from src.metric_tracker import MetricTracker, Mode


class TestMetricTracker:
    @staticmethod
    def test_init_metrics() -> None:
        arg_list = ["--no-save", "--no-visualize", "--num-examples=100", "--epochs=1"]
        args, _, _ = init_pipeline(arg_list)

        metrics = MetricTracker(args, {})

        assert metrics

    @staticmethod
    def test_one_batch_update(example_batch: SimpleNamespace) -> None:
        arg_list = ["--no-save", "--no-visualize", "--epochs=1"]
        args, _, _ = init_pipeline(arg_list)
        metrics = MetricTracker(args, {})

        tqdm_dict = metrics.batch_update(example_batch, 0, 1, Mode.TRAIN)

        for key in tqdm_dict:
            tqdm_dict[key] = round(tqdm_dict[key], 2)
        result = [round(metric.epoch_avg, 2) for metric in metrics.metric_data.values()]
        assert tqdm_dict == {"Loss": 0.21, "Accuracy": 0.67}
        assert result == [0.63, 2.0]

    @staticmethod
    def test_many_batch_update(example_batch: SimpleNamespace) -> None:
        arg_list = ["--no-save", "--no-visualize", "--epochs=1", "--log-interval=3"]
        args, _, _ = init_pipeline(arg_list)
        metrics = MetricTracker(args, {})
        num_batches = 4

        for i in range(num_batches):
            tqdm_dict = metrics.batch_update(example_batch, i, num_batches, Mode.TRAIN)

        for key in tqdm_dict:
            tqdm_dict[key] = round(tqdm_dict[key], 2)
        result = [round(metric.epoch_avg, 2) for metric in metrics.metric_data.values()]
        assert tqdm_dict == {"Loss": 0.21, "Accuracy": 0.67}
        assert all(metric.running_avg == 0.0 for metric in metrics.metric_data.values())
        assert result == [2.52, 8.0]

    @staticmethod
    def test_epoch_update(capsys: CaptureFixture, example_batch: SimpleNamespace) -> None:
        arg_list = ["--no-save", "--no-visualize", "--epochs=1", "--log-interval=3"]
        args, _, _ = init_pipeline(arg_list)
        metrics = MetricTracker(args, {})
        num_batches = 4
        for i in range(num_batches):
            _ = metrics.batch_update(example_batch, i, num_batches, Mode.TRAIN)

        metrics.epoch_update(Mode.TRAIN)

        captured = capsys.readouterr().out
        assert captured == "Mode.TRAIN Loss: 0.2100 Accuracy: 66.67% \n"

    @staticmethod
    def test_checkpoint_continue() -> None:
        pass
