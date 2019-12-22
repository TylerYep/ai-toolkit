import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR

from dataset import load_data
from models import BasicCNN
from test import test
import util


def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_steps = epoch * len(train_loader) + i

        if i % args.log_interval == 0:
            print((f'Train Epoch: {epoch} [{i * len(data)}/{len(train_loader.dataset)}'
                   f'({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'))
            writer.add_scalar('training loss', running_loss / args.log_interval, num_steps)
            running_loss = 0.0
        break
    return 0


def main():
    args = util.get_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = load_data(args)

    model = BasicCNN().to(device)
    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = util.load_checkpoint(args.checkpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    best_loss = np.inf
    run_name = util.get_run_name()
    writer = SummaryWriter(run_name)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch, writer)
        val_loss = test(args, model, device, test_loader)
        # scheduler.step()

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        util.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'epoch': epoch
        }, run_name, is_best)


if __name__ == '__main__':
    main()
