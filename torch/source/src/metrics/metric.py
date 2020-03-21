class Metric:
    def __init__(self):
        self.name = type(self).__name__
        self.init_val = 0.0
        self.epoch_avg = 0.0
        self.running_avg = 0.0
        self.num_examples = 0

    def formatted(self, computed_val):
        return f'{self.name}: {computed_val:.4f}'

    def reset(self):
        self.running_avg = 0.0

    def get_batch_result(self, log_interval, batch_size):
        return self.running_avg / (log_interval * batch_size)

    def get_epoch_result(self):
        return self.epoch_avg / self.num_examples
