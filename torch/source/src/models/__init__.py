import sys

from .rnn import BasicRNN
from .cnn import BasicCNN
from .dense import DenseNet
# from .unet import UNet
# from .efficient_net import EfficientNet


def get_model_initializer(model_name):
    ''' Retrieves class initializer from its string name. '''
    assert hasattr(sys.modules[__name__], model_name), \
        f'Model class {model_name} not found in models folder.'
    return getattr(sys.modules[__name__], model_name)
