import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
if torch.cuda.is_available():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import util
from args import init_pipeline
from util import Metrics
from dataset import load_train_data
from models import BasicCNN as Model
from viz import visualize


def update_metrics(metrics, loss, epoch, i, pbar, loader, accuracy, mode):
    metrics.update('epoch_loss', loss.item())
    metrics.update('running_loss', loss.item())
    metrics.update('epoch_acc', accuracy.item())
    metrics.update('running_acc', accuracy.item())

    if i % metrics.log_interval == 0:
        num_steps = (epoch-1) * len(loader) + i
        metrics.write(f'{mode} Loss', metrics.running_loss / metrics.log_interval, num_steps)
        metrics.write(f'{mode} Accuracy', metrics.running_acc / metrics.log_interval, num_steps)
        metrics.reset(['running_loss', 'running_acc'])

    pbar.set_postfix({'Loss': f'{loss.item():.5f}', 'Accuracy': accuracy.item()})
    pbar.update()


def get_results(metrics, data_loader, epoch, mode):
    print(f'{mode} Loss: {metrics.epoch_loss / len(data_loader):.4f}',
          f'Accuracy: {100 * metrics.epoch_acc / len(data_loader):.2f}%')
    metrics.write(f'{mode} Epoch Loss', metrics.epoch_loss / len(data_loader), epoch)
    metrics.write(f'{mode} Epoch Accuracy', metrics.epoch_acc / len(data_loader), epoch)
    return metrics.epoch_loss


def train_model(args, model, criterion, train_loader, epoch, device, metrics, optimizer=None):
    model.train()
    with tqdm(desc='Train Batch', total=len(train_loader), ncols=120) as pbar:
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if args.visualize:
                visualize(data, target)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(1) == target).float().mean()
            update_metrics(metrics, loss, epoch, i, pbar, train_loader, accuracy, mode='Train')

    epoch_result = get_results(metrics, val_loader, epoch, mode='Train')
    metrics.reset()
    return epoch_result


def validate_model(args, model, criterion, val_loader, epoch, device, metrics):
    model.eval()
    with torch.no_grad():
        with tqdm(desc='Val Batch', total=len(val_loader), ncols=120) as pbar:
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                accuracy = (output.argmax(1) == target).float().mean()
                update_metrics(metrics, loss, epoch, i, pbar, val_loader, accuracy, mode='Val')

    epoch_result = get_results(metrics, val_loader, epoch, mode='Val')
    metrics.reset()
    return epoch_result


def main():
    args, device = init_pipeline()

    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = F.nll_loss

    checkpoint = util.load_checkpoint(args.checkpoint, model, optimizer) if args.checkpoint else None
    run_name = checkpoint['run_name'] if checkpoint else util.get_run_name()
    start_epoch = checkpoint['epoch'] if checkpoint else 1

    metrics = Metrics(SummaryWriter(run_name),
                      ('epoch_loss', 'running_loss', 'epoch_acc', 'running_acc'),
                      args.log_interval)
    train_loader, val_loader = load_train_data(args)
    # summary(model, (1, 64, 64))

    best_loss = np.inf
    for epoch in range(start_epoch, args.epochs + 1):
        print(f'Epoch [{epoch}/{args.epochs}]')
        train_model(args, model, criterion, train_loader, epoch, device, metrics, optimizer)
        val_loss = validate_model(args, model, criterion, val_loader, epoch, device, metrics)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        util.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'run_name': run_name,
            'epoch': epoch
        }, run_name, is_best)


if __name__ == '__main__':
    main()
