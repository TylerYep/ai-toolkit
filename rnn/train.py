import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsummary

import util
from args import init_pipeline
from dataset import load_train_data
from metric_tracker import MetricTracker, Mode
from models import RNN as Model
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
    args, device = init_pipeline()
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, params = load_train_data(args)
    model = Model(*params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = util.load_checkpoint(args.checkpoint, model, optimizer)

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
    metric_checkpoint = checkpoint['metric_obj'] if checkpoint else {}
    metrics = MetricTracker(METRIC_NAMES, run_name, args.log_interval, **metric_checkpoint)

    best_loss = np.inf
    start_epoch = metrics.epoch + 1
    for epoch in range(start_epoch, start_epoch + args.epochs + 1):
        print(f'Epoch [{epoch}/{args.epochs}]')
        metrics.set_epoch(epoch)
        train_loss = train_and_validate(train_loader, metrics, run_name, Mode.TRAIN)
        val_loss = train_and_validate(val_loader, metrics, run_name, Mode.VAL)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        util.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'run_name': run_name,
            'metric_obj': metrics.json_repr()
        }, run_name, is_best)


if __name__ == '__main__':
    main()
