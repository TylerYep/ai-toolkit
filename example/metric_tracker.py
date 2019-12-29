from enum import Enum, unique
from torch.utils.tensorboard import SummaryWriter

from metrics import get_metric

@unique
class Mode(Enum):
    TRAIN = 'Train'
    VAL = 'Val'


class MetricTracker:
    def __init__(self, run_name, metric_names, log_interval):
        assert metric_names
        self.run_name = run_name
        self.writer = SummaryWriter(run_name)
        self.epoch = 1
        self.metric_names = metric_names
        self.primary_metric = metric_names[0]
        self.metric_data = {name: get_metric(name)() for name in self.metric_names}
        self.log_interval = log_interval
        self.num_examples = 0

    def __getattr__(self, name):
        return self.metric_data[name]

    def write(self, title, val, step_num):
        self.writer.add_scalar(title, val, step_num)

    def set_epoch(self, new_epoch):
        self.epoch = new_epoch

    def set_num_examples(self, num_examples):
        self.num_examples = num_examples

    def reset_all(self):
        for metric in self.metric_data:
            self.metric_data[metric].reset()

    def reset_hard(self):
        self.metric_data = {name: get_metric(name)() for name in self.metric_names}

    def update_all(self, val_dict):
        for metric in self.metric_data:
            self.metric_data[metric].update(val_dict)

    def write_all(self, num_steps, mode):
        for metric, metric_obj in self.metric_data.items():
            batch_result = metric_obj.get_batch_result(self.log_interval)
            self.write(f'{mode}_Batch_{metric}', batch_result, num_steps)

    def batch_update(self, i, data, loss, output, target, mode):
        accuracy = (output.argmax(1) == target).float().mean()
        val_dict = {'Loss': loss.item(), 'Accuracy': accuracy.item()}
        self.update_all(val_dict)
        num_steps = (self.epoch-1) * self.num_examples + i
        if i % self.log_interval == 0 and mode == Mode.TRAIN:
            if i > 0:
                self.write_all(num_steps, mode)
            self.reset_all()

        # if mode == Mode.VAL:
        #     for j in range(output.shape[0]):
        #         pred, ind = torch.max(output.data[j], dim=0)
        #         self.writer.add_image(f'{int(target.data[j])}/Pred:{ind}',
        #                                 data[j],
        #                                 num_steps)

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
