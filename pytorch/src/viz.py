from __future__ import annotations

from typing import Any, Iterator

import torch.nn as nn

from src import util
from src.args import Arguments, init_pipeline
from src.datasets import get_dataset_initializer
from src.metric_tracker import MetricTracker
from src.models import get_model_initializer
from src.visualizations import (
    compute_activations,
    create_class_visualization,
    make_fooling_image,
    show_saliency_maps,
    view_input,
)


def viz() -> None:
    args, device, checkpoint = init_pipeline()
    dataset_loader = get_dataset_initializer(args.dataset)
    train_loader, _, init_params = dataset_loader.load_train_data(args, device)
    init_params = checkpoint.get("model_init", init_params)
    model = get_model_initializer(args.model)(*init_params).to(device)
    util.load_state_dict(checkpoint, model)

    sample_loader = util.get_sample_loader(train_loader)
    visualize(args, model, sample_loader)
    visualize_trained(args, model, sample_loader)


def visualize(
    args: Arguments,
    model: nn.Module,
    loader: Iterator[Any],
    metrics: MetricTracker | None = None,
) -> None:
    if not args.no_visualize and metrics is not None:
        metrics.add_network(model, loader)

        run_name = metrics.run_name
        data, target = next(loader)
        view_input(data, target, metrics.class_labels, run_name)
        data, target = next(loader)
        compute_activations(model, data, target, metrics.class_labels, run_name)


def visualize_trained(
    args: Arguments,
    model: nn.Module,
    loader: Iterator[Any],
    metrics: MetricTracker | None = None,
) -> None:
    if not args.no_visualize and metrics is not None:
        run_name = metrics.run_name
        data, target = next(loader)
        make_fooling_image(
            model, data[5], target[5], metrics.class_labels, target[9], run_name
        )
        data, target = next(loader)
        show_saliency_maps(model, data, target, metrics.class_labels, run_name)
        data, target = next(loader)
        create_class_visualization(
            model, data, target[1], metrics.class_labels, run_name
        )


if __name__ == "__main__":
    viz()
