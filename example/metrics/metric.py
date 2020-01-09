class Metric:
    def __init__(self):
        self.name = type(self).__name__
        self.init_val = 0.0

    def formatted(self, computed_val):
        return f'{self.name}: {computed_val:.4f}'
