from efficientnet_pytorch import EfficientNet as ENet  # type: ignore[import-untyped]
from torch import nn


class EfficientNet(nn.Module):
    """Neural network"""

    def __init__(self, input_shape, num_classes=23, pretrained=True, b=0):
        super().__init__()
        self.input_shape = input_shape
        model_str = f"efficientnet-b{b}"
        self.efficient_net = (
            ENet.from_pretrained(model_str, num_classes=num_classes)
            if pretrained
            else ENet.from_name(model_str, num_classes=num_classes)
        )

    def forward(self, x):
        """Forward pass for your network."""
        x = self.efficient_net(x)
        return x
