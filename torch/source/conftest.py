''' conftest.py '''
from types import SimpleNamespace
import os
import shutil
import pytest
import torch


@pytest.fixture(autouse=True)
def reset_const():
    TEST_CHECKPOINT = os.path.join('checkpoints', 'TEST')
    if os.path.isdir(TEST_CHECKPOINT):
        shutil.rmtree(TEST_CHECKPOINT)


@pytest.fixture
def example_output():
    val_dict = SimpleNamespace(**{
        'loss': torch.tensor([0.2]),
        'output': torch.tensor([
            [1., 0, 0, 0, 0],
            [0.1, 0.7, 0.2, 0, 0],
            [0.1, 0.7, 0, 0, 0.8],
        ]),
        'target': torch.tensor([0, 1, 3]),
        'batch_size': 3
    })
    return val_dict


# def debug_spacing_issues(captured: str, expected: str):
#     ''' Helper method for debugging print differences. '''
#     print(len(captured), len(expected))
#     for i, captured_char in enumerate(captured):
#         if captured_char != expected[i]:
#             print("INCORRECT: ", i, captured_char, "vs", expected[i])
#         else:
#             print(" " * 10, i, captured_char, "vs", expected[i])
