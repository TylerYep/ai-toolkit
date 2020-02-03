import torchvision.models as models

from src import util
from src.args import init_pipeline
from src.dataset import load_train_data
from src.models import BasicCNN as Model

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
    view_input(data, target, class_labels, run_name)
    data, target = util.get_data_example(loader, device)
    compute_activations(model, data, target, class_labels, run_name)


def visualize_trained(model, loader, class_labels, device, run_name='', metrics=None):
    data, target = util.get_data_example(loader, device)
    make_fooling_image(model, data[5], target[5], class_labels, target[9], run_name)
    data, target = util.get_data_example(loader, device)
    show_saliency_maps(model, data, target, class_labels, run_name)
    data, target = util.get_data_example(loader, device)
    create_class_visualization(model, data, class_labels, target[1], run_name)


if __name__ == '__main__':
    main()
