import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

import util
from util import AverageMeter
from dataset import load_data
from models import BasicCNN
from test import test_model
from viz import visualize


def train_model(args, model, criterion, train_loader, optimizer, epoch, writer):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loss, running_loss = 0.0, 0.0
    with tqdm(desc='Batch', total=len(train_loader), ncols=120, position=1, leave=True) as pbar:
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if args.visualize:
                visualize(data, target)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            running_loss += loss.item()
            num_steps = epoch * len(train_loader) + i

            if i % args.log_interval == 0:
                writer.add_scalar('training loss', running_loss / args.log_interval, num_steps)
                running_loss = 0.0

            # pbar.set_postfix({'Loss': loss_meter.avg, 'Accuracy': acc_meter.avg})
            pbar.set_postfix({'Loss': f'{loss.item():.5f}', 'Accuracy': 0})
            pbar.update()

    return train_loss


def main():
    args = util.get_args()
    util.set_seed()

    ###
    model = BasicCNN()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    criterion = F.nll_loss
    ###

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = util.load_checkpoint(args.checkpoint, model, optimizer)
        start_epoch = checkpoint['epoch']

    run_name = util.get_run_name()
    writer = SummaryWriter(run_name)
    train_loader, val_loader, _ = load_data(args)

    best_loss = np.inf
    with tqdm(desc='Epoch', total=args.epochs + 1, ncols=120, position=0, leave=True) as pbar:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_model(args, model, criterion, train_loader, optimizer, epoch, writer)
            val_loss = test_model(args, model, criterion, val_loader)

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            util.save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'run_name': run_name,
                'epoch': epoch
            }, run_name, is_best)

            pbar.update()


if __name__ == '__main__':
    main()
