import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicCNN(nn.Module):
    """ Neural network """
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropx1 = nn.Dropout2d(0.25)
        self.dropx2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """ Forward pass for your feedback prediction network. """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropx1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropx2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def forward_with_activations(self, x):
        x = self.conv1(x)
        first_activation = x
        x = F.relu(x)
        second_activation = x
        x = self.conv2(x)
        third_activation = x
        x = F.max_pool2d(x, 2)
        fourth_activation = x
        x = self.dropx1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropx2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x, [first_activation, second_activation, third_activation, fourth_activation]
