""" conftest.py """
from types import SimpleNamespace

import pytest

import torch


@pytest.fixture
def example_batch():
    batch_size = 3
    val_dict = SimpleNamespace(
        **{
            "data": torch.zeros([batch_size, 1, 28, 28]),
            "loss": torch.tensor([0.21]),
            "output": torch.tensor(
                [[1.0, 0, 0, 0, 0], [0.1, 0.7, 0.2, 0, 0], [0.1, 0.7, 0, 0, 0.8],]
            ),
            "target": torch.tensor([0, 1, 3]),
            "batch_size": batch_size,
        }
    )
    return val_dict
