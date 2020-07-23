import json
import os
import random
import shutil
from argparse import Namespace
from typing import Any, Dict, Generator

import numpy as np
import torch
import torch.nn as nn


class Arguments:
    def __init__(self, args: Namespace):
        self.__dict__ = args

    def __repr__(self) -> str:
        return str(self.__dict__)


def load_args_from_json(args: Namespace) -> Arguments:
    filename = args.config
    found_json = os.path.join(args.config_dir, filename + ".json")
    if not os.path.isfile(found_json):
        found_json = os.path.join(args.save_dir, filename, "args.json")
    with open(found_json) as f:
        new_args = vars(args)
        new_args.update(json.load(f))
        return Arguments(new_args)


def get_run_name(args: Namespace) -> str:
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if args.checkpoint:
        return os.path.join(save_dir, args.checkpoint)

    if args.name:
        full_name = os.path.join(save_dir, args.name)
        if not os.path.isdir(full_name):
            os.makedirs(full_name)
        return full_name

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


def get_sample_loader(loader) -> Generator:
    """ Returns a generator that outputs a single batch of data. """
    sample_loader = iter(loader)
    while True:
        try:
            yield next(sample_loader)
        except StopIteration:
            sample_loader = iter(loader)


def set_rng_state(checkpoint: Dict[str, Any]) -> None:
    if checkpoint:
        random.setstate(checkpoint["rng_state"])
        np.random.set_state(checkpoint["np_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])


def save_checkpoint(state: Dict[str, Any], is_best: bool, run_name: str = "") -> None:
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


def load_checkpoint(checkpoint_path: str, use_best: bool = False) -> Dict[str, Any]:
    """ Loads torch checkpoint.
    Args:
        checkpoint_path: (string) filename which needs to be loaded
    """
    load_file = "model_best.pth.tar" if use_best else "checkpoint.pth.tar"
    return torch.load(os.path.join(checkpoint_path, load_file))


def load_state_dict(checkpoint: Dict, model: nn.Module, optimizer=None, scheduler=None):
    """ Loads model parameters (state_dict) from checkpoint. If optimizer or scheduler are
    provided, loads state_dict of optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: () checkpoint object
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if checkpoint:
        print("Loading checkpoint...")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
