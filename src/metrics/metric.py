from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class Metric:
    epoch_avg: float = 0
    running_avg: float = 0
    num_examples: int = 0

    def __post_init__(self) -> None:
        self.name = type(self).__name__

    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.4f}"

    @property
    def value(self) -> float:
        if self.num_examples == 0:
            return self.epoch_avg
        return self.epoch_avg / self.num_examples

    def update(self, val_dict: SimpleNamespace) -> float:
        raise NotImplementedError

    def batch_reset(self) -> None:
        self.running_avg = 0

    def epoch_reset(self) -> None:
        self.epoch_avg = 0
        self.running_avg = 0
        self.num_examples = 0

    def get_batch_result(self, batch_size: int, log_interval: int = 1) -> float:
        if log_interval <= 0 or batch_size <= 0:
            raise RuntimeError("log_interval and batch_size must be positive.")
        return self.running_avg / (log_interval * batch_size)
