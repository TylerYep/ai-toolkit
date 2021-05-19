from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

# Redefining here to avoid circular import
TensorDataLoader = DataLoader[Tuple[torch.Tensor, ...]]


def get_sample_loader(loader: TensorDataLoader) -> Iterator[Any]:
    """Returns a generator that outputs a single batch of data."""
    sample_loader = iter(loader)
    while True:
        try:
            yield next(sample_loader)
        except StopIteration:
            sample_loader = iter(loader)


def set_rng_state(checkpoint: dict[str, Any]) -> None:
    if checkpoint:
        random.setstate(checkpoint["rng_state"])
        np.random.set_state(checkpoint["np_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])


def save_checkpoint(state: dict[str, Any], is_best: bool, run_name: str = "") -> None:
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
    run_name_path = Path(run_name or state["run_name"])
    save_path = run_name_path / "checkpoint.pth.tar"
    torch.save(state, save_path)
    if is_best:
        print("Saving new model_best...\n")
        shutil.copyfile(save_path, run_name_path / "model_best.pth.tar")


def load_checkpoint(checkpoint_path: Path, use_best: bool = False) -> dict[str, Any]:
    """
    Loads torch checkpoint.
    Args:
        checkpoint_path: (string) filename which needs to be loaded
    """
    load_file = "model_best.pth.tar" if use_best else "checkpoint.pth.tar"
    checkpoint = torch.load(checkpoint_path / load_file)
    return cast(Dict[str, Any], checkpoint)


def load_state_dict(
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: lr_scheduler._LRScheduler | None = None,
) -> None:
    """
    Loads model parameters (state_dict) from checkpoint. If optimizer or scheduler are
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
