import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchsummary

import util
from util import Metrics, Mode
from args import init_pipeline
from dataset import load_train_data, INPUT_SHAPE
from models import BasicCNN as Model
from viz import visualize, compute_activations, compute_saliency

if torch.cuda.is_available():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


METRIC_NAMES = ('epoch_loss', 'running_loss', 'epoch_acc', 'running_acc')


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
    assert (data.grad[0: test_val] == 0.).all() and (data.grad[test_val+1:] == 0.).all()


def main():
    args, device = init_pipeline()
    criterion = F.nll_loss
    model = Model().to(device)
    torchsummary.summary(model, INPUT_SHAPE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = util.load_checkpoint(args.checkpoint, model, optimizer)

    def train_and_validate(loader, metrics, mode) -> float:
        if mode == Mode.TRAIN:
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        metrics.set_num_examples(len(loader))
        with tqdm(desc='', total=len(loader), ncols=120) as pbar:
            for i, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                should_visualize = args.visualize and i == 0 and metrics.epoch == 1

                if mode == Mode.TRAIN:
                    optimizer.zero_grad()

                    if should_visualize:
                        visualize(data, target, metrics.run_name)
                        compute_activations(model, data, metrics.run_name)
                        data.requires_grad_()

                output = model(data)
                loss = criterion(output, target)

                if mode == Mode.TRAIN:
                    loss.backward()
                    optimizer.step()

                    if should_visualize:
                        compute_saliency(data, metrics.run_name)

                update_metrics(metrics, pbar, i, data, loss, output, target, mode)
        return get_results(metrics, mode)

    train_loader, val_loader = load_train_data(args)
    verify_model(model, train_loader, optimizer, device)

    run_name = checkpoint['run_name'] if checkpoint else util.get_run_name(args)
    metrics = Metrics(run_name, METRIC_NAMES, args.log_interval)

    best_loss = np.inf
    start_epoch = checkpoint['epoch'] if checkpoint else 1
    for epoch in range(start_epoch, args.epochs + 1):
        print(f'Epoch [{epoch}/{args.epochs}]')
        metrics.set_epoch(epoch)
        train_loss = train_and_validate(train_loader, metrics, Mode.TRAIN)
        val_loss = train_and_validate(val_loader, metrics, Mode.VAL)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        util.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'run_name': run_name,
            'epoch': epoch
        }, run_name, is_best)


def update_metrics(metrics, pbar, i, data, loss, output, target, mode) -> None:
    accuracy = (output.argmax(1) == target).float().mean()
    metrics.update('epoch_loss', loss.item())
    metrics.update('running_loss', loss.item())
    metrics.update('epoch_acc', accuracy.item())
    metrics.update('running_acc', accuracy.item())

    num_steps = (metrics.epoch-1) * metrics.num_examples + i
    if i % metrics.log_interval == 0 and i > 0:
        metrics.write(f'{mode} Loss', metrics.running_loss / metrics.log_interval, num_steps)
        metrics.write(f'{mode} Accuracy', metrics.running_acc / metrics.log_interval, num_steps)
        metrics.reset(['running_loss', 'running_acc'])

    if mode == Mode.VAL:
        for j in range(output.shape[0]):
            pred, ind = torch.max(output.data[j], dim=0)
            metrics.writer.add_image(f'{int(target.data[j])}/Pred:{ind}',
                                     data[j],
                                     num_steps)

    pbar.set_postfix({'Loss': f'{loss.item():.5f}', 'Accuracy': accuracy.item()})
    pbar.update()


def get_results(metrics, mode) -> float:
    total_loss = metrics.epoch_loss / metrics.num_examples
    total_acc = 100. * metrics.epoch_acc / metrics.num_examples
    print(f'{mode} Loss: {total_loss:.4f} Accuracy: {total_acc:.2f}%')
    metrics.write(f'{mode} Epoch Loss', total_loss, metrics.epoch)
    metrics.write(f'{mode} Epoch Accuracy', total_acc, metrics.epoch)
    metrics.reset()
    return total_loss


if __name__ == '__main__':
    main()
