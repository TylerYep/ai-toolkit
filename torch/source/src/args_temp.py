from typing import List, Optional
import os
import argparse
import random
import numpy as np
import torch

from src import util


def init_pipeline(arg_list: Optional[List[str]] = None):
    ''' Pass in the empty list to skip argument parsing. '''
    set_random_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_parsed_arguments(arg_list)
    if args.config:
        # Update additional configs defined in the json file.
        args = util.load_args_from_json(args.config)

    checkpoint = {}
    if args.checkpoint:
        checkpoint_path = os.path.join(args.save_dir, args.checkpoint)
        checkpoint = util.load_checkpoint(checkpoint_path, args.use_best)
    return args, device, checkpoint


def get_parsed_arguments(arg_list):
    parser = argparse.ArgumentParser(description='PyTorch ML Pipeline')

    parser.add_argument('--batch-dim', type=int, default=$batch_dim, metavar='B',
                        help='batch dimension for training (default: 0)')

    parser.add_argument('--batch-size', type=int, default=128, metavar='B',
                        help='input batch size for training (default: 128)')

    parser.add_argument('--checkpoint', type=str, default='', metavar="CKPT",
                        help='for loading a checkpoint model')

    parser.add_argument('--config', type=str, default='',
                        help='use given config file as args: <checkpoints, configs>/<name>.json')

    parser.add_argument('--config-dir', type=str, default='configs',
                        help='config directory to use')

    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs to train (default: 100)')

    parser.add_argument('--gamma', type=float, default=0.7, metavar='G',
                        help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--img-dim', type=int, default=256, metavar='N',
                        help='set image size')

    parser.add_argument('--log-interval', type=int, default=10, metavar='NB',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--loss', type=str, default='$loss_fn', metavar='LOSS',
                        help='loss function to use')

    parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                        help='learning rate (default: 3e-3)')

    parser.add_argument('--metrics', nargs='+', type=str, default=['Loss', 'Accuracy'],
                        help='metrics in metrics/ folder to use during training (space-separated)')

    parser.add_argument('--model', type=str, default='$model', metavar='MODEL',
                        help='model architecture to use')

    parser.add_argument('--name', type=str, default='', metavar='NAME',
                        help='existing folder in checkpoint/ to save files to')

    parser.add_argument('--num-examples', type=int, default=None, metavar='N',
                        help='number of training examples')

    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot training examples')

    parser.add_argument('--no-save', action='store_true', default=False,
                        help='do not save any checkpoints')

    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='checkpoint directory to use')

    parser.add_argument('--scheduler', action='store_true', default=False,
                        help='use learning rate scheduler')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--use-best', action='store_true', default=False,
                        help='use checkpoint with best val metric rather than most recent')

    parser.add_argument('--no-verify', action='store_true', default=False,
                        help='do not perform model verification')

    parser.add_argument('--no-visualize', action='store_true', default=False,
                        help='do not save visualization files')

    return parser.parse_args(arg_list)


def set_random_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
