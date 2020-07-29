import json
import os
from enum import Enum, unique
from types import SimpleNamespace
from typing import Any, Dict, Generator, List, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.args import Arguments, get_run_name
from src.metrics import Metric, get_metric_initializer


@unique
class Mode(Enum):
    TRAIN, VAL, TEST = "Train", "Val", "Test"


class MetricTracker:
    def __init__(
        self, args: Arguments, checkpoint: Dict[str, Any], class_labels: Optional[List[str]] = None
    ) -> None:
        assert args.metrics
        self.run_name, self.writer = None, None
        if not args.no_save:
            self.run_name = checkpoint.get("run_name", get_run_name(args))
            self.writer = SummaryWriter(self.run_name)
            print(f"Storing checkpoints in: {self.run_name}\n")
            with open(os.path.join(self.run_name, "args.json"), "w") as f:
                json.dump(args.__dict__, f, indent=4)

        self.class_labels = [] if class_labels is None else class_labels

        metric_checkpoint = checkpoint.get("metric_obj", {})
        self.epoch = metric_checkpoint.get("epoch", 0)
        self.is_best = metric_checkpoint.get("is_best", True)
        self.metric_data = metric_checkpoint.get(
            "metric_data", {name: get_metric_initializer(name)() for name in args.metrics}
        )
        self.primary_metric = metric_checkpoint.get("primary_metric", args.metrics[0])
        self.end_epoch = self.epoch + args.epochs
        self.args = args
        self.prev_best = None

    def json_repr(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "metric_data": self.metric_data,
            "primary_metric": self.primary_metric,
            "is_best": self.is_best,
        }

    def __getitem__(self, name: str) -> Metric:
        return self.metric_data[name]

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, MetricTracker)
        for metric in self.metric_data:
            if (
                metric not in other.metric_data
                or self.metric_data[metric] != other.metric_data[metric]
            ):
                return False
        return True

    def next_epoch(self) -> None:
        self.epoch += 1
        print(f"Epoch [{self.epoch}/{self.end_epoch}]")

    def reset_hard(self) -> None:
        for metric in self.metric_data.values():
            metric.epoch_reset()

    def batch_update(
        self, val_dict: SimpleNamespace, i: int, num_batches: int, mode: Mode
    ) -> Dict[str, float]:
        assert torch.isfinite(val_dict.loss).all(), "The loss returned in training is NaN or inf."
        tqdm_dict = {}
        for metric, metric_obj in self.metric_data.items():
            metric_obj.update(val_dict)
            tqdm_dict[metric] = metric_obj.value

        num_steps = (self.epoch - 1) * num_batches + i
        # Only reset batch statistics after log_interval batches
        if i > 0 and i % self.args.log_interval == 0:
            if mode == Mode.TRAIN:
                # Write batch to tensorboard
                for metric, metric_obj in self.metric_data.items():
                    batch_result = metric_obj.get_batch_result(
                        val_dict.batch_size, self.args.log_interval
                    )
                    self.write(f"{Mode.TRAIN}_Batch_{metric}", batch_result, num_steps)

            for metric in self.metric_data.values():
                metric.batch_reset()

        if mode == Mode.VAL and not self.args.no_visualize:
            if hasattr(val_dict.data, "size") and len(val_dict.data.size()) == 4:  # (N, C, H, W)
                self.add_images(val_dict, num_steps)
        return tqdm_dict

    def epoch_update(self, mode: Mode) -> None:
        result_str = f"{mode} "
        for metric, metric_obj in self.metric_data.items():
            self.write(f"{mode}_Epoch_{metric}", metric_obj.value, self.epoch)
            if mode == Mode.VAL and metric == self.primary_metric:
                self.is_best = self.prev_best is None or metric_obj.value < self.prev_best
                if self.is_best:
                    self.prev_best = metric_obj.value
            result_str += f"{metric_obj} "
        print(result_str)

    def add_network(self, model: nn.Module, loader: Generator) -> None:
        if self.writer is not None:
            data, _ = next(loader)  # type: ignore
            self.writer.add_graph(model, data)

    def write(self, title: str, val: float, step_num: int):
        if self.writer is not None:
            self.writer.add_scalar(title, val, step_num)

    def add_images(self, val_dict: SimpleNamespace, num_steps: int) -> None:
        if self.writer is not None and self.class_labels:
            data, output, target = val_dict.data, val_dict.output, val_dict.target
            for j in range(output.shape[0]):
                _, pred_ind = torch.max(output.detach()[j], dim=0)
                target_ind = int(target.detach()[j])
                pred_class = self.class_labels[int(pred_ind)]
                target_class = self.class_labels[target_ind]
                self.writer.add_image(f"{target_class}/Predicted_{pred_class}", data[j], num_steps)