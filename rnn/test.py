import torch
import torch.nn.functional as F

import util
from args import init_pipeline
from dataset import load_test_data
from models import BasicCNN as Model

if torch.cuda.is_available():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def test_model(test_loader, model, device, criterion):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        with tqdm(desc='Test', total=len(test_loader), ncols=120) as pbar:
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
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')


def main():
    args, device = init_pipeline()
    criterion = F.nll_loss
    test_loader, params = load_test_data(args)
    model = Model(*params).to(device)
    util.load_checkpoint(args.checkpoint, model)

    test_model(test_loader, model, device, criterion)


if __name__ == '__main__':
    main()
