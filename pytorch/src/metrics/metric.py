from dataclasses import dataclass


@dataclass
class Metric:
    epoch_avg: float = 0.0
    running_avg: float = 0.0
    num_examples: int = 0

    def __post_init__(self):
        self.name = type(self).__name__

    def __repr__(self):
        return f"{self.name}: {self.value:.4f}"

    def update(self, val_dict):
        raise NotImplementedError

    def batch_reset(self):
        self.running_avg = 0.0

    def epoch_reset(self):
        self.epoch_avg = 0.0
        self.running_avg = 0.0
        self.num_examples = 0

    def get_batch_result(self, batch_size, log_interval=1):
        assert log_interval > 0
        assert batch_size > 0
        return self.running_avg / (log_interval * batch_size)

    @property
    def value(self):
        if self.num_examples == 0:
            return self.epoch_avg
        return self.epoch_avg / self.num_examples
