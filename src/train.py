from __future__ import annotations

import random
import sys
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src import util
from src.args import Arguments, init_pipeline
from src.datasets import TensorDataLoader, get_dataset_initializer
from src.losses import get_loss_initializer
from src.metric_tracker import MetricTracker, Mode
from src.models import get_model_initializer
from src.verify import verify_model
from src.viz import visualize, visualize_trained

if "google.colab" in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def train_and_validate(
    args: Arguments,
    model: nn.Module,
    loader: TensorDataLoader,
    optimizer: optim.Optimizer | None,
    criterion: nn.Module,
    metrics: MetricTracker,
    mode: Mode,
) -> None:
    if mode == Mode.TRAIN:
        model.train()
    else:
        model.eval()

    torch.set_grad_enabled(mode == Mode.TRAIN)
    metrics.reset_hard()
    num_batches = len(loader)
    with tqdm(desc=str(mode), total=num_batches, ncols=120) as pbar:
        for i, (data, target) in enumerate(loader):
            # If you have multiple optimizers, use model.zero_grad().
            # If you want to freeze layers, use optimizer.zero_grad().
            if mode == Mode.TRAIN and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            if isinstance(data, (list, tuple)):
                output = model(*data)
                batch_size = data[0].size(args.batch_dim)
            else:
                output = model(data)
                batch_size = data.size(args.batch_dim)

            loss = criterion(output, target)
            if mode == Mode.TRAIN and optimizer is not None:
                loss.backward()
                optimizer.step()

            val_dict = {
                "data": data,
                "loss": loss,
                "output": output,
                "target": target,
                "batch_size": batch_size,
            }
            tqdm_dict = metrics.batch_update(
                SimpleNamespace(**val_dict), i, num_batches, mode
            )
            pbar.set_postfix(tqdm_dict)
            pbar.update()
    metrics.epoch_update(mode)


def get_optimizer(args: Arguments, model: nn.Module) -> optim.Optimizer:
    params = filter(lambda p: p.requires_grad, model.parameters())
    return optim.AdamW(params, lr=args.lr)


def get_scheduler(
    args: Arguments, optimizer: optim.Optimizer
) -> lr_scheduler._LRScheduler:
    return lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)


def load_model(
    args: Arguments,
    device: torch.device,
    init_params: tuple[Any, ...],
    loader: Iterator[Any],
) -> tuple[nn.Module, nn.Module, optim.Optimizer, lr_scheduler._LRScheduler | None]:
    criterion = get_loss_initializer(args.loss)()
    model = get_model_initializer(args.model)(*init_params).to(device)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer) if args.scheduler else None
    verify_model(args, model, loader, optimizer, criterion, device)
    return model, criterion, optimizer, scheduler


def train(*arg_list: str) -> MetricTracker:
    args, device, checkpoint = init_pipeline(*arg_list)
    dataset_loader = get_dataset_initializer(args.dataset)
    train_loader, val_loader, init_params = dataset_loader.load_train_data(args, device)
    sample_loader = util.get_sample_loader(train_loader)
    model, criterion, optimizer, scheduler = load_model(
        args, device, init_params, sample_loader
    )
    util.load_state_dict(checkpoint, model, optimizer, scheduler)
    metrics = MetricTracker(args, checkpoint, dataset_loader.CLASS_LABELS)
    visualize(args, model, sample_loader, metrics)

    util.set_rng_state(checkpoint)
    for _ in range(args.epochs):
        metrics.next_epoch()
        train_and_validate(
            args, model, train_loader, optimizer, criterion, metrics, Mode.TRAIN
        )
        train_and_validate(args, model, val_loader, None, criterion, metrics, Mode.VAL)
        if scheduler is not None:
            scheduler.step()

        if not args.no_save:
            checkpoint_dict = {
                "model_init": init_params,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    None if scheduler is None else scheduler.state_dict()
                ),
                "rng_state": random.getstate(),
                "np_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "run_name": metrics.run_name,
                "metric_obj": metrics.json_repr(),
            }
            util.save_checkpoint(checkpoint_dict, metrics.is_best)

    torch.set_grad_enabled(True)
    visualize_trained(args, model, sample_loader, metrics)
    return metrics
