""" Imports all Metric objects. """
import sys
from typing import Any

from .accuracy import Accuracy
from .dice import Dice
from .f1_score import F1Score
from .iou import IoU
from .loss import Loss
from .metric import Metric


def get_metric_initializer(metric_name: str) -> Any:
    """ Retrieves class initializer from its string name. """
    if not hasattr(sys.modules[__name__], metric_name):
        raise RuntimeError(f"Metric {metric_name} not found in metrics folder.")
    return getattr(sys.modules[__name__], metric_name)


__all__ = (
    "Accuracy",
    "Dice",
    "F1Score",
    "IoU",
    "Loss",
    "Metric",
    "get_metric_initializer",
)
