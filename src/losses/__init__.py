""" Imports all Loss functions. """
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice import DiceLoss
from .focal import FocalLoss


def get_loss_initializer(loss_fn: str) -> Any:
    """Retrieves class initializer from its string name."""
    if loss_fn == "nn.CrossEntropyLoss":
        return nn.CrossEntropyLoss
    if loss_fn in ("F.nll_loss", "nn.NLLLoss"):
        return nn.NLLLoss
    if not hasattr(sys.modules[__name__], loss_fn):
        raise RuntimeError(f"Metric {loss_fn} not found in metrics folder.")
    return getattr(sys.modules[__name__], loss_fn)


__all__ = ("DiceLoss", "FocalLoss", "get_loss_initializer")
