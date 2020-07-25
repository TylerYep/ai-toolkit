from src import util
from src.args import init_pipeline
from src.datasets import get_dataset_initializer
from src.models import get_model_initializer
from src.visualizations import (
    compute_activations,
    create_class_visualization,
    make_fooling_image,
    show_saliency_maps,
    view_input,
)


def viz():
    args, device, checkpoint = init_pipeline()
    dataset_loader = get_dataset_initializer(args.dataset)
    train_loader, _, init_params = dataset_loader.load_train_data(args, device)
    init_params = checkpoint.get("model_init", init_params)
    model = get_model_initializer(args.model)(*init_params).to(device)
    util.load_state_dict(checkpoint, model)

    sample_loader = iter(train_loader)
    visualize(args, model, sample_loader)
    visualize_trained(args, model, sample_loader)


def visualize(args, model, loader, metrics=None):
    if not args.no_visualize and (metrics is None or metrics.class_labels):
        metrics.add_network(model, loader)

        run_name = metrics.run_name
        data, target = next(loader)
        view_input(data, target, metrics.class_labels, run_name)
        data, target = next(loader)
        compute_activations(model, data, target, metrics.class_labels, run_name)


def visualize_trained(args, model, loader, metrics=None):
    if not args.no_visualize and (metrics is None or metrics.class_labels):
        run_name = metrics.run_name
        data, target = next(loader)
        make_fooling_image(model, data[5], target[5], metrics.class_labels, target[9], run_name)
        data, target = next(loader)
        show_saliency_maps(model, data, target, metrics.class_labels, run_name)
        data, target = next(loader)
        create_class_visualization(model, data, metrics.class_labels, target[1], run_name)


if __name__ == "__main__":
    viz()
