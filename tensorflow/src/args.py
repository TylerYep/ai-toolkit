import argparse
import random

import numpy as np

import tensorflow as tf
from src import util


def init_pipeline(arg_list=None):
    """ Pass in the empty list to skip argument parsing. """
    set_random_seeds()
    args = get_parsed_arguments(arg_list)
    if args.config:
        args = util.json_to_args(args.config)
    checkpoint = util.load_checkpoint(args.checkpoint)
    return args, checkpoint


def get_parsed_arguments(arg_list):
    parser = argparse.ArgumentParser(description="PyTorch ML Pipeline")

    # fmt: off
    parser.add_argument("--batch-size", type=int, default=128, metavar="N",
                        help="input batch size for training (default: 128)")

    parser.add_argument("--checkpoint", type=str, default="",
                        help="for loading a checkpoint model")

    parser.add_argument("--config", type=str, default="",
                        help="use given config file as args: <checkpoints, configs>/<name>.json")

    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="number of epochs to train (default: 100)")

    parser.add_argument("--gamma", type=float, default=0.7, metavar="M",
                        help="Learning rate step gamma (default: 0.7)")

    parser.add_argument("--img-dim", type=int, default=256, metavar="N",
                        help="set image size")

    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")

    parser.add_argument("--loss", type=str, default="$loss_fn", metavar="N",
                        help="loss function to use")

    parser.add_argument("--lr", type=float, default=3e-3, metavar="LR",
                        help="learning rate (default: 3e-3)")

    parser.add_argument("--metrics", nargs="+", type=str, default=["Loss", "Accuracy"],
                        help="metrics in metrics/ folder to use during training (space-separated)")

    parser.add_argument("--model", type=str, default="$model", metavar="N",
                        help="model architecture to use")

    parser.add_argument("--name", type=str, default="",
                        help="folder to save files to checkpoint/")

    parser.add_argument("--num-examples", type=int, default=None, metavar="N",
                        help="number of training examples")

    parser.add_argument("--plot", action="store_true", default=False,
                        help="plot training examples")

    parser.add_argument("--scheduler", action="store_true", default=False,
                        help="use learning rate scheduler")

    parser.add_argument("--test-batch-size", type=int, default=128, metavar="N",
                        help="input batch size for testing (default: 128)")

    parser.add_argument("--visualize", action="store_true", default=True,
                        help="save visualization files")
    # fmt: on

    return parser.parse_args(arg_list)


def set_random_seeds():
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)
