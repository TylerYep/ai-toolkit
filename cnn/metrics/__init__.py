''' Imports all Metric objects. '''
from typing import Any
import sys

from .loss import Loss
from .accuracy import Accuracy

def get_metric(metric_name: str) -> Any:
    ''' Retrieves class initializer from its string name. '''
    return getattr(sys.modules[__name__], metric_name)
