import sys

from .dataset_cnn import DatasetCNN
from .dataset_lstm import DatasetLSTM
from .dataset_penn import DatasetPenn
from .dataset_rnn import DatasetRNN


def get_dataset_initializer(model_name):
    """ Retrieves class initializer from its string name. """
    assert hasattr(
        sys.modules[__name__], model_name
    ), f"Dataset class {model_name} not found in datasets/"
    return getattr(sys.modules[__name__], model_name)()
