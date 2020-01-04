class Metric:
    def __init__(self):
        self.name = type(self).__name__

    def formatted(self, computed_val):
        return f'{self.name}: {computed_val:.4f}'
