from __future__ import annotations

import sys
from typing import Type, cast

import torch.nn as nn

from .cnn import BasicCNN
from .dense import DenseNet
from .lstm import BasicLSTM
from .maskrcnn import MaskRCNN
from .rnn import BasicRNN

# from .unet import UNet
# from .efficient_net import EfficientNet


def get_model_initializer(model_name: str) -> type[nn.Module]:
    """Retrieves class initializer from its string name."""
    if not hasattr(sys.modules[__name__], model_name):
        raise RuntimeError(f"Model class {model_name} not found in models/")
    return cast(Type[nn.Module], getattr(sys.modules[__name__], model_name))


__all__ = (
    "BasicCNN",
    "DenseNet",
    "BasicLSTM",
    "MaskRCNN",
    "BasicRNN",
    "get_model_initializer",
)
