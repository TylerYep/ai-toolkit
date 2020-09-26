import json
import os
import random
import shutil
from argparse import Namespace
from typing import Any, Dict

import numpy as np

SAVE_DIR = "checkpoints"
CONFIG_DIR = "configs"


def set_rng_state(checkpoint: Any) -> None:
    if checkpoint:
        random.setstate(checkpoint["rng_state"])
        np.random.set_state(checkpoint["np_rng_state"])
        # torch.set_rng_state(checkpoint["torch_rng_state"])


def save_checkpoint(state: Dict[str, Any], is_best: bool, run_name: str = "") -> None:
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best is True, also saves best.pth.tar
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as
        epoch, optimizer_state_dict
        run_name: (string) folder where parameters are to be saved
        is_best: (bool) True if it is the best model seen till now
    """
    print("Saving checkpoint...\n")
    # run_name = run_name if run_name else state["run_name"]
    # save_path = os.path.join(run_name, "checkpoint.pth.tar")
    # torch.save(state, save_path)
    # if is_best:
    #     print("Saving new model_best...\n")
    #     shutil.copyfile(save_path, os.path.join(run_name, "model_best.pth.tar"))


def load_checkpoint(checkpoint_name: str, use_best: bool = False) -> Dict[str, Any]:
    """
    Loads torch checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
    """
    if not checkpoint_name:
        return {}
    print("Loading checkpoint...")
    return {}
    # load_file = "model_best.pth.tar" if use_best else "checkpoint.pth.tar"
    # return torch.load(os.path.join(SAVE_DIR, checkpoint_name, load_file))
