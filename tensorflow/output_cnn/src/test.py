import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

from src import util
from src.args import init_pipeline
from src.dataset import load_test_data, INPUT_SHAPE
from src.models import BasicCNN as Model

if 'google.colab' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def test_model(test_loader, model, device, criterion):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)


def main():
    args, device, checkpoint = init_pipeline()
    criterion = F.nll_loss
    test_loader = load_test_data(args)
    init_params = checkpoint.get('model_init', {})
    model = Model(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    model.summary()

    test_model(test_loader, model, device, criterion)


if __name__ == '__main__':
    main()
