from typing import Dict, Any
from argparse import Namespace
import os
import shutil
import torch
import torch.nn as nn

SAVE_DIR = 'checkpoints'


def get_run_name(args: Namespace, save_dir: str = SAVE_DIR) -> str:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.name:
        result = args.name
    else:
        dirlist = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
        dirlist.sort()
        dirlist.sort(key=lambda k: (len(k), k))  # Sort alphabetically but by length
        if not dirlist:
            result = 'A'
        else:
            last_run_char = dirlist[-1][-1]
            if last_run_char == 'Z':
                result = 'A' * (len(dirlist[-1])+1)
            else:
                result = dirlist[-1][:-1] + chr(ord(last_run_char) + 1)
    out_dir = os.path.join(save_dir, result)
    os.makedirs(out_dir)
    return out_dir


def save_checkpoint(state: Dict[str, Any], run_name: str, is_best: bool) -> None:
    """ Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best is True, also saves best.pth.tar
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as
        epoch, optimizer_state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    print('Saving checkpoint...')
    save_path = os.path.join(run_name, 'checkpoint.pth.tar')
    torch.save(state, save_path)
    if is_best:
        print('Saving new model_best...')
        shutil.copyfile(save_path, os.path.join(run_name, 'model_best.pth.tar'))


def load_checkpoint(checkpoint_name: str):
    """ Loads torch checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
    """
    if not checkpoint_name:
        return {}
    print('Loading checkpoint...')
    return torch.load(os.path.join(SAVE_DIR, checkpoint_name, 'checkpoint.pth.tar'))


def load_state_dict(checkpoint: Dict, model: nn.Module, optimizer=None) -> Dict[str, Any]:
    """ Loads model parameters (state_dict) from checkpoint. If optimizer is provided,
    loads state_dict of optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: () checkpoint object
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    print(checkpoint)
    if not checkpoint:
        return {}
    torch.set_rng_state(checkpoint['rng_state'])
    if model is not None:
        model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
