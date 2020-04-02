from typing import Any, Dict
from types import SimpleNamespace
from enum import Enum, unique
import torch
from torch.utils.tensorboard import SummaryWriter

from src import util
from src.metrics import get_metric_initializer
from src.dataset import CLASS_LABELS


@unique
class Mode(Enum):
    TRAIN, VAL, TEST = 'Train', 'Val', 'Test'


class MetricTracker:
    def __init__(self, args, checkpoint):
        assert args.metrics
        self.run_name = checkpoint.get('run_name', util.get_run_name(args))
        self.writer = SummaryWriter(self.run_name)
        print(f'Storing checkpoints in: {self.run_name}\n')

        self.log_interval = args.log_interval
        self.primary_metric = args.metrics[0]

        metric_checkpoint = checkpoint.get('metric_obj', {})
        self.epoch = metric_checkpoint.get('epoch', 0)
        self.metric_data = metric_checkpoint.get('metric_data', self.init_metrics(args.metrics))
        self.best_metric = metric_checkpoint.get('best_metric',
                                                 self.metric_data[self.primary_metric].init_val)
        self.end_epoch = self.epoch + args.epochs

    @staticmethod
    def init_metrics(metric_names):
        return {name: get_metric_initializer(name)() for name in metric_names}

    def json_repr(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'metric_data': self.metric_data,
            'best_metric': self.best_metric
        }

    def __getattr__(self, name):
        return self.metric_data[name]

    def add_network(self, model, loader):
        data, _ = next(loader)
        self.writer.add_graph(model, data)

    def update_best_metric(self, val_loss) -> bool:
        is_best = val_loss < self.best_metric
        self.best_metric = min(val_loss, self.best_metric)
        return is_best

    def write(self, title: str, val: float, step_num: int):
        self.writer.add_scalar(title, val, step_num)

    def next_epoch(self):
        self.epoch += 1
        print(f'Epoch [{self.epoch}/{self.end_epoch}]')

    def reset_all(self):
        for metric in self.metric_data:
            self.metric_data[metric].reset()

    def reset_hard(self):
        self.metric_data = self.init_metrics(self.metric_data.keys())

    def update_all(self, val_dict):
        ret_dict = {}
        for metric, metric_obj in self.metric_data.items():
            metric_obj.update(val_dict)
            ret_dict[metric] = metric_obj.get_epoch_result()
        return ret_dict

    def write_all(self, num_steps, mode, batch_size):
        for metric, metric_obj in self.metric_data.items():
            batch_result = metric_obj.get_batch_result(batch_size, self.log_interval)
            self.write(f'{mode}_Batch_{metric}', batch_result, num_steps)

    def add_images(self, val_dict, num_steps):
        data, output, target = val_dict.data, val_dict.output, val_dict.target
        for j in range(output.shape[0]):
            _, pred_ind = torch.max(output.detach()[j], dim=0)
            target_ind = int(target.detach()[j])
            pred_class = CLASS_LABELS[pred_ind]
            target_class = CLASS_LABELS[target_ind]
            self.writer.add_image(f'{target_class}/Predicted_{pred_class}', data[j], num_steps)

    def batch_update(self, i, num_batches, data, loss, output, target, mode):
        batch_size = data.shape[0]
        names = ('data', 'loss', 'output', 'target', 'batch_size')
        variables = (data, loss, output, target, batch_size)
        val_dict = SimpleNamespace(**dict(zip(names, variables)))

        tqdm_dict = self.update_all(val_dict)
        num_steps = (self.epoch - 1) * num_batches + i
        if mode == Mode.TRAIN and i % self.log_interval == 0:
            if i > 0:
                self.write_all(num_steps, mode, batch_size)
            self.reset_all()
        elif mode == Mode.VAL:
            if len(data.size()) == 4:  # (N, C, H, W)
                self.add_images(val_dict, num_steps)
        return tqdm_dict

    def get_epoch_results(self, mode) -> float:
        result_str = ''
        ret_val = None
        for metric, metric_obj in self.metric_data.items():
            epoch_result = metric_obj.get_epoch_result()
            if metric == self.primary_metric:
                ret_val = epoch_result
            result_str += str(metric_obj)
            self.write(f'{mode}_Epoch_{metric}', epoch_result, self.epoch)

        print(f'{mode} {result_str}')
        self.reset_hard()
        return ret_val
