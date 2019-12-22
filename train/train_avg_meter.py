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
from util import get_args

import numpy as np
from tqdm import tqdm

def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0


##
    loss_meter = AverageMeter()  # utility for tracking loss / accuracy
    acc_meter = AverageMeter()

    label_arr, pred_arr = [], []
    pbar = tqdm(len(train_loader))
##


    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    ##
        loss_meter.update(loss.item(), 1000)
        pred_npy = torch.round(output).detach().numpy()
        label_npy = target.detach().numpy()
        acc = np.mean(pred_npy == label_npy)
        acc_meter.update(acc, 1000)

        label_arr.append(label_npy)
        pred_arr.append(pred_npy)

        pbar.set_postfix({'Loss': loss_meter.avg, 'Accuracy': acc_meter.avg})
        pbar.update()
        ##

        running_loss += loss.item()
        num_steps = epoch * len(train_loader) + i

        # if i % args.log_interval == 0:
        #     print((f'Train Epoch: {epoch} [{i * len(data)}/{len(train_loader.dataset)}'
        #            f'({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'))
        #     writer.add_scalar('training loss', running_loss / args.log_interval, num_steps)
        #     running_loss = 0.0


import os
import shutil
def get_run_name(save_dir='checkpoints'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dirlist = sorted([f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))])
    dirlist.sort(key=lambda k: (len(k), k)) # Sort alphabetically but by length
    if len(dirlist) == 0:
        result = 'A'
    else:
        last_run_char = dirlist[-1][-1]
        if last_run_char == 'Z':
            result = 'A' * (len(dirlist[-1])+1)
        else:
            result = dirlist[-1][:-1] + chr(ord(last_run_char) + 1)
    out_dir = os.path.join(save_dir, result)
    os.makedirs(out_dir)
    return out_dir


def save_checkpoint(state, save_path, is_best):
    ''' Saves a copy of the model (+ properties) to filesystem. '''
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(folder, 'model_best.pth.tar'))


def main():
    args = get_args()

    run_name = get_run_name()
    save_path = os.path.join(run_name, 'checkpoint.pth.tar')
    writer = SummaryWriter(run_name)

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = load_data(args)

    model = BasicCNN().to(device)
    start_epoch = 1
    if args.checkpoint_path != '':
        checkpoint = torch.load(args.checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(start_epoch, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader)
        # scheduler.step()

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'epoch': epoch
        }, save_path, is_best)


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
