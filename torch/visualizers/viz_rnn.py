import torchvision.models as models

from src import util
from src.args import init_pipeline
from src.dataset import load_train_data
from src.models import BasicRNN as Model

from src.visualizations import *


def viz():
    args, device, checkpoint = init_pipeline()
    train_loader, _, init_params = load_train_data(args, device)
    model = Model(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    # model = models.resnet18(pretrained=True)

    visualize(model, train_loader)
    visualize_trained(model, train_loader)


def visualize(model, loader, run_name='', metrics=None):
    data, target = next(iter(loader))
    pass


def visualize_trained(model, loader, run_name='', metrics=None):
    data, target = next(iter(loader))
    pass


if __name__ == '__main__':
    viz()
