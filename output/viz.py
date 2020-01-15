import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models

import util
from args import init_pipeline
from dataset import load_train_data, INPUT_SHAPE
from models import BasicCNN as Model

from visualizations import *


def main():
    args, device, checkpoint = init_pipeline()
    train_loader, _, class_labels, init_params = load_train_data(args)
    model = Model(*init_params).to(device)
    # model = models.resnet18(pretrained=True)

    visualize(model, train_loader, class_labels, None, device)


# TODO saliency, class_viz, fooling needs trained model
def visualize(model, train_loader, class_labels, optimizer, device, run_name=''):
    data, target = next(iter(train_loader))
    view_input(data, target, class_labels, run_name)
    data, target = next(iter(train_loader))
    compute_activations(model, data, target, class_labels, run_name)

    data, target = next(iter(train_loader))
    make_fooling_image(model, data[5], target[5], class_labels, target[9], run_name)
    data, target = next(iter(train_loader))
    show_saliency_maps(model, data, target, class_labels, run_name)
    data, target = next(iter(train_loader))
    create_class_visualization(model, data, class_labels, target[1], run_name)


if __name__ == '__main__':
    main()
