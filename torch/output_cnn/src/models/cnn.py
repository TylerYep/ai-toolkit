import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicCNN(nn.Module):
    """ Neural network """
    def __init__(self):
        super().__init__()
        self.input_shape = torch.Size((1, 28, 28))
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """ Forward pass for your feedback prediction network. """
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.max_pool2d(out, 2)
        out = self.dropout1(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

    def forward_with_activations(self, x):
        out = self.conv1(x)
        first_activation = out
        out = F.relu(out)
        second_activation = out
        out = self.conv2(out)
        third_activation = out
        out = F.max_pool2d(out, 2)
        fourth_activation = out
        out = self.dropout1(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out, [first_activation, second_activation, third_activation, fourth_activation]
