import json
import os
import random
import shutil
from argparse import Namespace
from typing import Any, Dict

import numpy as np

import torch

SAVE_DIR = "checkpoints"
CONFIG_DIR = "configs"


class Arguments:
    def __init__(self, args):
        self.__dict__ = args

    def __repr__(self):
        return str(self.__dict__)


def json_to_args(filename):
    found_json = os.path.join(CONFIG_DIR, filename + ".json")
    if not os.path.isfile(found_json):
        found_json = os.path.join(SAVE_DIR, filename, "args.json")
    with open(found_json) as f:
        return Arguments(json.load(f))


def get_run_name(args: Namespace, save_dir: str = SAVE_DIR) -> str:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if args.checkpoint:
        return os.path.join(save_dir, args.checkpoint)

    if args.name:
        return os.path.join(save_dir, args.name)

    dirlist = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
    dirlist.sort()
    dirlist.sort(key=lambda k: (len(k), k))  # Sort alphabetically but by length
    if not dirlist:
        result = "A"
    else:
        last_run_char = dirlist[-1][-1]
        if last_run_char == "Z":
            result = "A" * (len(dirlist[-1]) + 1)
        else:
            result = dirlist[-1][:-1] + chr(ord(last_run_char) + 1)
    out_dir = os.path.join(save_dir, result)
    os.makedirs(out_dir)
    return out_dir


def set_rng_state(checkpoint):
    if checkpoint:
        random.setstate(checkpoint["rng_state"])
        np.random.set_state(checkpoint["np_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])


def save_checkpoint(state: Dict[str, Any], is_best: bool, run_name: str = ""):
    """ Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best is True, also saves best.pth.tar
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as
        epoch, optimizer_state_dict
        run_name: (string) folder where parameters are to be saved
        is_best: (bool) True if it is the best model seen till now
    """
    print("Saving checkpoint...\n")
    run_name = run_name if run_name else state["run_name"]
    save_path = os.path.join(run_name, "checkpoint.pth.tar")
    torch.save(state, save_path)
    if is_best:
        print("Saving new model_best...\n")
        shutil.copyfile(save_path, os.path.join(run_name, "model_best.pth.tar"))


def load_checkpoint(checkpoint_name: str, use_best: bool = False) -> Dict[str, Any]:
    """ Loads torch checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
    """
    if not checkpoint_name:
        return {}
    print("Loading checkpoint...")
    load_file = "model_best.pth.tar" if use_best else "checkpoint.pth.tar"
    return torch.load(os.path.join(SAVE_DIR, checkpoint_name, load_file))
