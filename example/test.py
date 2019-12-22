import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from dataset import load_data
from util import get_args


def test(args, model, device, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print((f'\nTest set: Average loss: {test_loss:.4f}, '
           f'Accuracy: {correct}/{len(test_loader.dataset)} '
           f'({100. * correct / len(test_loader.dataset):.0f}%)\n'))
    return test_loss


def main():
    args = get_args()
    checkpoint = torch.load(args.checkpoint_path)

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    _, test_loader = load_data(args)
    model = BasicCNN().to(device)
    model.load_state_dict(checkpoint['state_dict']) # TODO

    test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
