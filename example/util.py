from typing import Dict, Any
from enum import Enum, unique
from argparse import Namespace
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

SAVE_DIR = 'checkpoints'


class Metrics:
    def __init__(self, run_name, metric_names, log_interval=10):
        self.run_name = run_name
        self.writer = SummaryWriter(run_name)
        self.epoch = 1
        self.metric_names = metric_names
        self.metric_data = {name: 0.0 for name in metric_names}
        self.log_interval = log_interval
        self.num_examples = 0

    def __getattr__(self, name) -> float:
        return self.metric_data[name]

    def set_epoch(self, new_epoch):
        self.epoch = new_epoch

    def reset(self, metric_names=None):
        if metric_names is None:
            metric_names = self.metric_names

        for name in metric_names:
            self.metric_data[name] = 0.0

    def update(self, name, val, n=1):
        self.metric_data[name] += val * n

    def write(self, title, val, step_num):
        self.writer.add_scalar(title, val, step_num)

    def set_num_examples(self, num_examples):
        self.num_examples = num_examples


def get_run_name(args: Namespace, save_dir: str = SAVE_DIR) -> str:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.name:
        result = args.name
    else:
        dirlist = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
        dirlist.sort()
        dirlist.sort(key=lambda k: (len(k), k))  # Sort alphabetically but by length
        if len(dirlist) == 0:
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
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.
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


def load_checkpoint(checkpoint_name: str, model: nn.Module, optimizer=None) -> Dict[str, Any]:
    """ Loads model parameters (state_dict) from file_path. If optimizer is provided,
    loads state_dict of optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not checkpoint_name:
        return {}
    print('Loading checkpoint...')
    checkpoint = torch.load(os.path.join(SAVE_DIR, checkpoint_name, 'checkpoint.pth.tar'))
    torch.set_rng_state(checkpoint['rng_state'])
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


@unique
class Mode(Enum):
    TRAIN = 'Train'
    VAL = 'Val'
