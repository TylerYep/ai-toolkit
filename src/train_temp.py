import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsummary

import util
from args import init_pipeline
from dataset import load_train_data, INPUT_SHAPE
from metric_tracker import MetricTracker, Mode
from models import $model as Model
from viz import visualize

if torch.cuda.is_available():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


METRIC_NAMES = ['Loss', 'Accuracy']


def verify_model(model, loader, optimizer, device, test_val=2):
    model.eval()
    torch.set_grad_enabled(True)
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    data.requires_grad_()

    output = model(data)
    loss = output[test_val].sum()
    loss.backward()

    assert loss.data != 0
    assert (data.grad[test_val] != 0).any()
    assert (data.grad[:test_val] == 0.).all() and (data.grad[test_val+1:] == 0.).all()


def main():
    args, device, checkpoint = init_pipeline()
    criterion = $loss_fn
    train_loader, val_loader, class_labels, init_params = load_train_data(args)
    model = Model(*init_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    util.load_state_dict(checkpoint, model, optimizer)
    torchsummary.summary(model, INPUT_SHAPE)

    def train_and_validate(loader, metrics, run_name, mode) -> float:
        if mode == Mode.TRAIN:
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        metrics.set_num_examples(len(loader))
        with tqdm(desc=str(mode), total=len(loader), ncols=120) as pbar:
            for i, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)

                if mode == Mode.TRAIN:
                    optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, target)

                if mode == Mode.TRAIN:
                    loss.backward()
                    optimizer.step()

                tqdm_dict = metrics.batch_update(i, data, loss, output, target, mode)
                pbar.set_postfix(tqdm_dict)
                pbar.update()

        return metrics.get_epoch_results(mode)

    verify_model(model, train_loader, optimizer, device)
    run_name = checkpoint['run_name'] if checkpoint else util.get_run_name(args)

    if args.visualize:
        visualize(model, train_loader, class_labels, optimizer, device, run_name)

    metric_checkpoint = checkpoint['metric_obj'] if checkpoint else {}
    metrics = MetricTracker(METRIC_NAMES, run_name, args.log_interval, **metric_checkpoint)
    util.set_rng_state(checkpoint)

    start_epoch = metrics.epoch + 1
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f'Epoch [{epoch}/{start_epoch + args.epochs - 1}]')
        metrics.set_epoch(epoch)
        train_loss = train_and_validate(train_loader, metrics, run_name, Mode.TRAIN)
        val_loss = train_and_validate(val_loader, metrics, run_name, Mode.VAL)

        is_best = metrics.update_best_metric(val_loss)
        util.save_checkpoint({
            'model_init': init_params,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': random.getstate(),
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'run_name': run_name,
            'metric_obj': metrics.json_repr()
        }, run_name, is_best)


if __name__ == '__main__':
    main()
