import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet as EffNet

class EfficientNet(nn.Module):
    """ Neural network """
    def __init__(self, num_classes=23, pretrained=True, b=0):
        super().__init__()
        self.input_shape = torch.Size((1, 64, 64))
        model_str = f'efficientnet-b{b}'
        self.efficient_net = EffNet.from_pretrained(model_str, num_classes=num_classes) \
                if pretrained else EffNet.from_name(model_str, num_classes=num_classes)

    def forward(self, x):
        """ Forward pass for your network. """
        out = self.efficient_net(x)
        return out
