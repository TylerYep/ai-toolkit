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
from models import BasicCNN
from viz import visualize


def update_metrics(metrics, args, loss, epoch, i, writer, pbar, loader, accuracy, mode='train'):
    metrics.update('epoch_loss', loss.item())
    metrics.update('running_loss', loss.item())
    metrics.update('epoch_acc', accuracy.item())
    metrics.update('running_acc', accuracy.item())

    if i % args.log_interval == 0:
        num_steps = (epoch-1) * len(loader) + i
        writer.add_scalar(f'{mode} Loss', metrics.running_loss / args.log_interval, num_steps)
        writer.add_scalar(f'{mode} Accuracy', metrics.running_acc / args.log_interval, num_steps)
        metrics.reset(['running_loss', 'running_acc'])

    pbar.set_postfix({'Loss': f'{loss.item():.5f}', 'Accuracy': accuracy.item()})
    pbar.update()


def train_model(args, model, criterion, train_loader, optimizer, epoch, writer, device):
    model.train()
    # summary(model, (1, 64, 64))
    metrics = Metrics(['epoch_loss', 'running_loss', 'epoch_acc', 'running_acc'])
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

            update_metrics(metrics, args, loss, epoch, i, writer, pbar, train_loader, accuracy)

    print(f'Train Loss: {metrics.epoch_loss / len(train_loader):.4f}',
          f'Accuracy: {100 * metrics.epoch_acc / len(train_loader):.2f}%')
    writer.add_scalar('training epoch loss', metrics.epoch_loss / len(train_loader), epoch)
    writer.add_scalar('training epoch accuracy', metrics.epoch_acc / len(train_loader), epoch)
    return metrics.epoch_loss


def validate_model(args, model, criterion, val_loader, epoch, writer, device):
    model.eval()
    metrics = Metrics(['epoch_loss', 'running_loss', 'epoch_acc', 'running_acc'])
    with torch.no_grad():
        with tqdm(desc='Val Batch', total=len(val_loader), ncols=120) as pbar:
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                accuracy = (output.argmax(1) == target).float().mean()
                update_metrics(metrics, args, loss, epoch, i, writer, pbar, val_loader, accuracy)

    print(f'Val Loss: {metrics.epoch_loss / len(val_loader):.4f}',
          f'Accuracy: {100 * metrics.epoch_acc / len(val_loader):.2f}%')
    writer.add_scalar('val epoch loss', metrics.epoch_loss / len(val_loader), epoch)
    writer.add_scalar('val epoch accuracy', metrics.epoch_acc / len(val_loader), epoch)
    return metrics.epoch_loss


def main():
    args, device = init_pipeline()

    model = BasicCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = F.nll_loss

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = util.load_checkpoint(args.checkpoint, model, optimizer)
        start_epoch = checkpoint['epoch']

    run_name = util.get_run_name()
    writer = SummaryWriter(run_name)
    train_loader, val_loader = load_train_data(args)

    best_loss = np.inf
    for epoch in range(start_epoch, args.epochs + 1):
        print(f'Epoch [{epoch}/{args.epochs}]')
        train_loss = train_model(args, model, criterion, train_loader, optimizer, epoch, writer, device)
        val_loss = validate_model(args, model, criterion, val_loader, epoch, writer, device)

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
