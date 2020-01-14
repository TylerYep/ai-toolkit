import argparse
import random
import numpy as np
import torch

import util

def init_pipeline():
    set_random_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='PyTorch ML Pipeline')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')

    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')

    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--checkpoint', type=str, default='',
                        help='for loading a checkpoint model')

    parser.add_argument('--name', type=str, default='',
                        help='folder to save files to checkpoint/')

    parser.add_argument('--visualize', action='store_true', default=True,
                        help='save visualization files ')

    args = parser.parse_args()
    checkpoint = util.load_checkpoint(args.checkpoint)

    return args, device, checkpoint


def set_random_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
