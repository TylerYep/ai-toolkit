from src import util
from src.args import init_pipeline
from src.dataset import load_train_data
from src.models import get_model_initializer

from src.visualizations import *


def viz():
    args, device, checkpoint = init_pipeline()
    train_loader, _, init_params = load_train_data(args, device)
    init_params = checkpoint.get('model_init', init_params)
    model = get_model_initializer(args.model)(*init_params).to(device)
    util.load_state_dict(checkpoint, model)

    visualize(model, train_loader)
    visualize_trained(model, train_loader)


def visualize(model, loader, run_name=''):
    data, target = next(loader)
    pass


def visualize_trained(model, loader, run_name=''):
    data, target = next(loader)
    pass


if __name__ == '__main__':
    viz()
