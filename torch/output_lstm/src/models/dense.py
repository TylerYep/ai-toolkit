import torch.nn as nn


class DenseNet(nn.Module):
    """ Neural network """
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        hidden_sizes = [128, 64]
        output_size = 10
        self.model = nn.Sequential(
            nn.Linear(784, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """ Forward pass for your feedback prediction network. """
        x = x.reshape((-1, 784))
        return self.model(x)

    def forward_with_activations(self, x):
        pass
