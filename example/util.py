import os
import shutil
import argparse
import torch
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')

    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')

    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--checkpoint', type=str, default='',
                        help='for loading a checkpoint model')

    parser.add_argument('--visualize', action='store_true', default=False,
                        help='input batch size for testing (default: False)')

    return parser.parse_args()


def set_seed():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_run_name(save_dir='checkpoints'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dirlist = sorted([f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))])
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


def save_checkpoint(state, run_name, is_best):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best is True, also saves best.pth.tar
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as
        epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    save_path = os.path.join(run_name, 'checkpoint.pth.tar')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(run_name, 'model_best.pth.tar'))


def load_checkpoint(checkpoint_run, model, optimizer=None):
    """ Loads model parameters (state_dict) from file_path. If optimizer is provided,
    loads state_dict of optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    checkpoint = torch.load(os.path.join('checkpoints', checkpoint_run, 'checkpoint.pth.tar'))
    torch.set_rng_state(checkpoint['rng_state'])
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
