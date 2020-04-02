class Metric:
    def __init__(self):
        self.name = type(self).__name__
        self.init_val = 0.0
        self.epoch_avg = 0.0
        self.running_avg = 0.0
        self.num_examples = 0

    def __repr__(self):
        return f'{self.name}: {self.get_epoch_result():.4f} '

    def reset(self):
        self.running_avg = 0.0

    def get_batch_result(self, batch_size, log_interval=1):
        assert log_interval > 0
        assert batch_size > 0
        return self.running_avg / (log_interval * batch_size)

    def get_epoch_result(self):
        assert self.num_examples > 0
        return self.epoch_avg / self.num_examples
