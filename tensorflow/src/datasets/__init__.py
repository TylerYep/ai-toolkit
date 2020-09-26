import sys
from typing import Any

from .dataset_cnn import DatasetCNN


def get_dataset_initializer(dataset_name: str) -> Any:
    """ Retrieves class initializer from its string name. """
    if not hasattr(sys.modules[__name__], dataset_name):
        raise RuntimeError(f"Dataset class {dataset_name} not found in datasets/")
    return getattr(sys.modules[__name__], dataset_name)()
