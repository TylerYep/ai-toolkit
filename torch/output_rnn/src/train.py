import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src import util
from src.args import init_pipeline
from src.dataset import load_train_data
from src.losses import get_loss_initializer
from src.metric_tracker import MetricTracker, Mode
from src.models import get_model_initializer
from src.verify import verify_model
from src.viz import visualize, visualize_trained

if 'google.colab' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def train_and_validate(args, model, loader, optimizer, criterion, metrics, mode):
    model.train() if mode == Mode.TRAIN else model.eval()
    torch.set_grad_enabled(mode == Mode.TRAIN)

    metrics.set_num_batches(len(loader))
    with tqdm(desc=str(mode), total=len(loader), ncols=120) as pbar:
        for i, (data, target) in enumerate(loader):
            if mode == Mode.TRAIN:
                optimizer.zero_grad()

            output = model(*data) if isinstance(data, (list, tuple)) else model(data)
            loss = criterion(output, target)
            if mode == Mode.TRAIN:
                loss.backward()
                optimizer.step()

            tqdm_dict = metrics.batch_update(i, data, loss, output, target, mode)
            pbar.set_postfix(tqdm_dict)
            pbar.update()

    return metrics.get_epoch_results(mode)


def init_metrics(args, checkpoint):
    run_name = checkpoint.get('run_name', util.get_run_name(args))
    print(f'Storing checkpoints in: {run_name}\n')
    metric_checkpoint = checkpoint.get('metric_obj', {})
    metrics = MetricTracker(run_name, args.log_interval, **metric_checkpoint)
    return run_name, metrics


def get_optimizer_schedulers(args, model):
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer) if args.scheduler else None
    return optimizer, scheduler


def load_model(args, device, checkpoint, init_params, train_loader):
    criterion = get_loss_initializer(args.loss)()
    model = get_model_initializer(args.model)(*init_params).to(device)
    assert model.input_shape, 'Model should have input_shape as an attribute'

    optimizer, scheduler = get_optimizer_schedulers(args, model)
    verify_model(model, train_loader, optimizer, criterion, device)
    util.load_state_dict(checkpoint, model, optimizer, scheduler)
    return model, criterion, optimizer, scheduler


def train(arg_list=None):
    args, device, checkpoint = init_pipeline(arg_list)
    train_loader, val_loader, init_params = load_train_data(args, device)
    model, criterion, optimizer, scheduler = load_model(args, device, checkpoint,
                                                        init_params, train_loader)
    run_name, metrics = init_metrics(args, checkpoint)
    if args.visualize:
        metrics.add_network(model, train_loader)
        visualize(model, train_loader, run_name)

    util.set_rng_state(checkpoint)
    start_epoch = metrics.epoch + 1
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f'Epoch [{epoch}/{start_epoch + args.epochs - 1}]')
        metrics.next_epoch()
        train_and_validate(args, model, train_loader, optimizer, criterion, metrics, Mode.TRAIN)
        val_loss = train_and_validate(args, model, val_loader, None, criterion, metrics, Mode.VAL)

        if args.scheduler:
            scheduler.step(val_loss)
        is_best = metrics.update_best_metric(val_loss)
        util.save_checkpoint({
            'model_init': init_params,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if args.scheduler else None,
            'rng_state': random.getstate(),
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'run_name': run_name,
            'metric_obj': metrics.json_repr()
        }, run_name, is_best)

    if args.visualize:
        visualize_trained(model, train_loader, run_name)

    return val_loss
