from src import util
from src.args import init_pipeline
from src.dataset import load_train_data, CLASS_LABELS
from src.models import get_model_initializer

from src.visualizations import view_input, compute_activations, make_fooling_image, \
    show_saliency_maps, create_class_visualization


def viz():
    args, device, checkpoint = init_pipeline()
    train_loader, _, init_params = load_train_data(args, device)
    init_params = checkpoint.get('model_init', init_params)
    model = get_model_initializer(args.model)(*init_params).to(device)
    util.load_state_dict(checkpoint, model)

    visualize(model, train_loader)
    visualize_trained(model, train_loader)


def visualize(model, loader, run_name='', metrics=None):
    data, target = next(loader)
    view_input(data, target, CLASS_LABELS, run_name)
    data, target = next(loader)
    compute_activations(model, data, target, CLASS_LABELS, run_name)


def visualize_trained(model, loader, run_name='', metrics=None):
    data, target = next(loader)
    make_fooling_image(model, data[5], target[5], CLASS_LABELS, target[9], run_name)
    data, target = next(loader)
    show_saliency_maps(model, data, target, CLASS_LABELS, run_name)
    data, target = next(loader)
    create_class_visualization(model, data, CLASS_LABELS, target[1], run_name)


if __name__ == '__main__':
    viz()
