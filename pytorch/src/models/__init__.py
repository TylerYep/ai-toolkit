import sys

from .cnn import BasicCNN
from .dense import DenseNet
from .lstm import BasicLSTM
from .maskrcnn import MaskRCNN
from .rnn import BasicRNN

# from .unet import UNet
# from .efficient_net import EfficientNet


def get_model_initializer(model_name):
    """ Retrieves class initializer from its string name. """
    assert hasattr(
        sys.modules[__name__], model_name
    ), f"Model class {model_name} not found in models/"
    return getattr(sys.modules[__name__], model_name)
