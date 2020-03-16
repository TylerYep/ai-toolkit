import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

from src import util
from src.args import init_pipeline
from src.dataset import load_test_data
from src.losses import get_loss_initializer
from src.models import get_model_initializer

if 'google.colab' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def test_model(test_loader, model, criterion):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        with tqdm(desc='Test', total=len(test_loader), ncols=120) as pbar:
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                pbar.update()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f},',
          f'Accuracy: {correct}/{len(test_loader.dataset)}',
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')


def test(arg_list=None):
    args, device, checkpoint = init_pipeline(arg_list)
    criterion = get_loss_initializer(args.loss)
    test_loader = load_test_data(args, device)
    init_params = checkpoint.get('model_init', [])
    model = get_model_initializer(args.model)(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    torchsummary.summary(model, model.input_shape)

    test_model(test_loader, model, criterion)
