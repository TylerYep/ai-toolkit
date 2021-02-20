from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from src import util

# import tensorflow as tf


@dataclass
class Arguments:
    batch_dim: int
    batch_size: int
    checkpoint: str
    config: str
    config_dir: str
    dataset: str
    epochs: int
    gamma: float
    img_dim: int
    log_interval: int
    loss: str
    lr: float
    metrics: str
    model: str
    name: str
    no_save: bool
    no_verify: bool
    no_visualize: bool
    num_examples: int
    num_workers: int
    plot: bool
    save_dir: str
    scheduler: bool
    test_batch_size: int
    use_best: bool


def get_parsed_arguments(arg_list: list[str] | None) -> Arguments:
    # fmt: off
    parser = argparse.ArgumentParser(description="PyTorch ML Pipeline")

    parser.add_argument("--batch-dim", type=int, default=0, metavar="B",
                        help="batch dimension for training (default: 0)")

    parser.add_argument("--batch-size", type=int, default=128, metavar="B",
                        help="input batch size for training (default: 128)")

    parser.add_argument("--checkpoint", type=str, default="", metavar="CKPT",
                        help="for loading a checkpoint model")

    parser.add_argument("--config", type=str, default="",
                        help="config file as args: <checkpoints, configs>/<name>.json")

    parser.add_argument("--config-dir", type=str, default="configs",
                        help="config directory to use")

    parser.add_argument("--dataset", type=str, default="DatasetCNN",
                        help="dataset in datasets/ folder to use")

    parser.add_argument("--epochs", type=int, default=100, metavar="E",
                        help="number of epochs to train (default: 100)")

    parser.add_argument("--gamma", type=float, default=0.7, metavar="G",
                        help="Learning rate step gamma (default: 0.7)")

    parser.add_argument("--img-dim", type=int, default=256, metavar="N",
                        help="set image size")

    parser.add_argument("--log-interval", type=int, default=10, metavar="NB",
                        help="how many batches to wait before logging training status")

    parser.add_argument("--loss", type=str, default="nn.CrossEntropyLoss",
                        metavar="LOSS", help="loss function to use")

    parser.add_argument("--lr", type=float, default=3e-3, metavar="LR",
                        help="learning rate (default: 3e-3)")

    parser.add_argument("--metrics", nargs="+", type=str, default=["Loss", "Accuracy"],
                        help="metrics to use during training (space-separated)")

    parser.add_argument("--model", type=str, default="BasicRNN", metavar="MODEL",
                        help="model architecture to use")

    parser.add_argument("--name", type=str, default="", metavar="NAME",
                        help="existing folder in checkpoint/ to save files to")

    parser.add_argument("--no-save", action="store_true", default=False,
                        help="do not save any checkpoints")

    parser.add_argument("--no-verify", action="store_true", default=False,
                        help="do not perform model verification")

    parser.add_argument("--no-visualize", action="store_true", default=False,
                        help="do not save visualization files")

    parser.add_argument("--num-examples", type=int, default=None, metavar="N",
                        help="number of training examples")

    parser.add_argument("--num-workers", type=int, default=0,
                        help="number of workers to use when training on GPU")

    parser.add_argument("--plot", action="store_true", default=False,
                        help="plot training examples")

    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="checkpoint directory to use")

    parser.add_argument("--scheduler", action="store_true", default=False,
                        help="use learning rate scheduler")

    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)")

    parser.add_argument("--use-best", action="store_true", default=False,
                        help="use best val metric checkpoint (default: most recent)")
    # fmt: on

    namespace = parser.parse_args(arg_list)
    return Arguments(**vars(namespace))


def load_args_from_json(args: Arguments) -> None:
    """ Update additional configs defined in the json file. """
    filename = args.config
    found_json = os.path.join(args.config_dir, filename + ".json")
    if not os.path.isfile(found_json):
        found_json = os.path.join(args.save_dir, filename, "args.json")
    with open(found_json) as f:
        arg_dict = json.load(f)
        for key, val in arg_dict.items():
            if not hasattr(args, key):
                raise KeyError(f"Not a valid argument for Arguments: {key}")
            setattr(args, key, val)


def init_pipeline(
    arg_list: list[str] | None = None,
) -> tuple[Arguments, dict[str, Any]]:
    """ Pass in the empty list to skip argument parsing. """
    set_random_seeds()
    args = get_parsed_arguments(arg_list)
    if args.config:
        load_args_from_json(args)

    checkpoint: dict[str, Any] = {}
    if args.checkpoint:
        checkpoint_path = os.path.join(args.save_dir, args.checkpoint)
        checkpoint = util.load_checkpoint(checkpoint_path, args.use_best)
    return args, checkpoint


def set_random_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
