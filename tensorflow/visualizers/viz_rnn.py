import torchvision.models as models

from src import util
from src.args import init_pipeline
from src.dataset import load_train_data
from src.models import BasicRNN as Model

from src.visualizations import *


def main():
    args, device, checkpoint = init_pipeline()
    train_loader, _, class_labels, init_params = load_train_data(args)
    model = Model(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    # model = models.resnet18(pretrained=True)

    visualize(model, train_loader, class_labels, device)
    visualize_trained(model, train_loader, class_labels, device)


def visualize(model, loader, class_labels, device, run_name='', metrics=None):
    data, target = util.get_data_example(loader, device)
    pass


def visualize_trained(model, loader, class_labels, device, run_name='', metrics=None):
    data, target = util.get_data_example(loader, device)
    pass


if __name__ == '__main__':
    main()
