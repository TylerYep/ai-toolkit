import torch
import torch.nn.functional as F
from tqdm import tqdm

import util
from dataset import load_data
from models import BasicCNN


def test_model(args, model, criterion, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_loss, correct = 0, 0
    with torch.no_grad():
        with tqdm(desc='Batch', total=len(test_loader), ncols=120, position=0) as pbar:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                pbar.update()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f},',
          f'Accuracy: {correct}/{len(test_loader.dataset)}',
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return test_loss


def main():
    args = util.get_args()
    util.set_seed()

    _, _, test_loader = load_data(args)
    model = BasicCNN()
    if args.checkpoint != '':
        util.load_checkpoint(args.checkpoint, model)

    criterion = F.nll_loss
    test_model(args, model, criterion, test_loader)


if __name__ == '__main__':
    main()
