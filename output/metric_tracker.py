from typing import Any, Dict
from enum import Enum, unique
import torch
from torch.utils.tensorboard import SummaryWriter

from metrics import get_metric


@unique
class Mode(Enum):
    TRAIN = 'Train'
    VAL = 'Val'
    TEST = 'Test'


class MetricTracker:
    def __init__(self,
                 metric_names,
                 run_name,
                 log_interval,
                 epoch=0,
                 num_examples=0,
                 metric_data=None,
                 best_metric=None):
        assert metric_names
        self.writer = SummaryWriter(run_name)
        self.epoch = epoch
        self.log_interval = log_interval
        self.num_examples = num_examples
        self.metric_names = metric_names
        self.primary_metric = metric_names[0]
        self.metric_data = metric_data if metric_data else \
                           {name: get_metric(name)() for name in self.metric_names}
        self.best_metric = best_metric if best_metric else \
                           self.metric_data[self.primary_metric].init_val

    def json_repr(self) -> Dict[str, Any]:
        return {'epoch': self.epoch,
                'metric_data': self.metric_data,
                'num_examples': self.num_examples,
                'best_metric': self.best_metric}

    def __getattr__(self, name):
        return self.metric_data[name]

    def update_best_metric(self, val_loss) -> bool:
        self.best_metric = min(val_loss, self.best_metric)
        is_best = val_loss < self.best_metric
        return is_best

    def write(self, title: str, val: float, step_num: int):
        self.writer.add_scalar(title, val, step_num)

    def set_epoch(self, new_epoch):
        self.epoch = new_epoch

    def set_num_examples(self, num_examples: int):
        self.num_examples = num_examples

    def reset_all(self):
        for metric in self.metric_data:
            self.metric_data[metric].reset()

    def reset_hard(self):
        self.metric_data = {name: get_metric(name)() for name in self.metric_names}

    def update_all(self, val_dict):
        ret_dict = {}
        for metric in self.metric_data:
            ret_dict[metric] = self.metric_data[metric].update(val_dict)
        return ret_dict

    def write_all(self, num_steps, mode):
        for metric, metric_obj in self.metric_data.items():
            batch_result = metric_obj.get_batch_result(self.log_interval)
            self.write(f'{mode}_Batch_{metric}', batch_result, num_steps)

    # Public Methods
    def batch_update(self, i, data, loss, output, target, mode):
        names = ('data', 'loss', 'output', 'target')
        variables = (data, loss, output, target)
        val_dict = dict(zip(names, variables))

        ret_dict = self.update_all(val_dict)
        num_steps = (self.epoch-1) * self.num_examples + i
        if mode == Mode.TRAIN and i % self.log_interval == 0:
            if i > 0:
                self.write_all(num_steps, mode)
            self.reset_all()

        # TODO make image its own metric, two types of metrics, qual vs quant
        # elif mode == Mode.VAL:
        #     for j in range(output.shape[0]):
        #         _, ind = torch.max(output.data[j], dim=0)
        #         self.writer.add_image(f'{int(target.data[j])}/Pred:{ind}',
        #                               data[j],
        #                               num_steps)
        return ret_dict

    def get_epoch_results(self, mode) -> float:
        result_str = ''
        for metric, metric_obj in self.metric_data.items():
            epoch_result = metric_obj.get_epoch_result(self.num_examples)
            result_str += f'{metric_obj.formatted(epoch_result)} '
            self.write(f'{mode}_Epoch_{metric}', epoch_result, self.epoch)

        print(f'{mode} {result_str}')
        ret_val = self.metric_data[self.primary_metric].get_epoch_result(self.num_examples)
        self.reset_hard()
        return ret_val
