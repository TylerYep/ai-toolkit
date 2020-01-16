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
    util.load_state_dict(checkpoint, model)
    # model = models.resnet18(pretrained=True)

    visualize(model, train_loader, class_labels, device)
    visualize_trained(model, train_loader, class_labels, device)


def visualize(model, train_loader, class_labels, device, metrics=None, run_name=''):
    data, target = util.get_data_example(train_loader, device)


def visualize_trained(model, train_loader, class_labels, device, metrics=None, run_name=''):
    data, target = util.get_data_example(train_loader, device)


if __name__ == '__main__':
    main()
