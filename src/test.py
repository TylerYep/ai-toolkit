from __future__ import annotations

import sys

import torch
import torch.nn as nn

from src import util
from src.args import Arguments, init_pipeline
from src.datasets import TensorDataLoader, get_dataset_initializer
from src.losses import get_loss_initializer
from src.models import get_model_initializer
from src.verify import model_summary

if "google.colab" in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def test_model(
    args: Arguments,
    model: nn.Module,
    test_loader: TensorDataLoader,
    criterion: nn.Module,
) -> None:
    model.eval()
    test_loss, correct = 0.0, 0.0
    test_len = len(test_loader)
    with torch.no_grad(), tqdm(desc="Test", total=test_len, ncols=120) as pbar:
        for data, target in test_loader:
            if isinstance(data, (list, tuple)):
                output = model(*data)
                batch_size = data[0].size(args.batch_dim)
            else:
                output = model(data)
                batch_size = data.size(args.batch_dim)
            loss = criterion(output, target)
            test_loss += loss.item() * batch_size
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.update()

    test_loss /= test_len
    print(
        f"\nTest set: Average loss: {test_loss:.4f},",
        f"Accuracy: {correct}/{test_len} ({100. * correct / test_len:.2f}%)\n",
    )


def test(*arg_list: str) -> None:
    args, device, checkpoint = init_pipeline(*arg_list)
    criterion = get_loss_initializer(args.loss)()
    test_loader = get_dataset_initializer(args.dataset).load_test_data(args, device)
    init_params = checkpoint.get("model_init", [])
    model = get_model_initializer(args.model)(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    sample_loader = util.get_sample_loader(test_loader)
    model_summary(args, model, sample_loader)

    test_model(args, model, test_loader, criterion)
