import os
import shutil
import argparse
import torch


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

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--checkpoint', type=str, default='',
                        help='for loading a checkpoint model')

    return parser.parse_args()


def get_run_name(save_dir='checkpoints'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dirlist = sorted([f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))])
    dirlist.sort(key=lambda k: (len(k), k)) # Sort alphabetically but by length
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
    ''' Saves a copy of the model (+ properties) to filesystem. '''
    save_path = os.path.join(run_name, 'checkpoint.pth.tar')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(run_name, 'model_best.pth.tar'))


def load_checkpoint(checkpoint_run):
    checkpoint = torch.load(os.path.join('checkpoints', checkpoint_run, 'checkpoint.pth.tar'))
    torch.set_rng_state(checkpoint['rng_state'])
    return checkpoint
