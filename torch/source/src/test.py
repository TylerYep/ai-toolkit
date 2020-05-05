import sys

import torch
from src import util
from src.args import init_pipeline
from src.dataset import load_test_data
from src.losses import get_loss_initializer
from src.models import get_model_initializer
from src.verify import model_summary

if "google.colab" in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def test_model(args, test_loader, model, criterion):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        with tqdm(desc="Test", total=len(test_loader), ncols=120) as pbar:
            for data, target in test_loader:
                output = model(*data) if isinstance(data, (list, tuple)) else model(data)
                loss = criterion(output, target)
                batch_size = (data[0] if isinstance(data, (list, tuple)) else data).size(args.batch_dim)
                test_loss += loss.item() * batch_size
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                pbar.update()

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f},",
        f"Accuracy: {correct}/{len(test_loader.dataset)}",
        f"({100. * correct / len(test_loader.dataset):.2f}%)\n",
    )


def test(arg_list=None):
    args, device, checkpoint = init_pipeline(arg_list)
    criterion = get_loss_initializer(args.loss)()
    test_loader = load_test_data(args, device)
    init_params = checkpoint.get("model_init", [])
    model = get_model_initializer(args.model)(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    sample_loader = util.get_sample_loader(test_loader)
    model_summary(args, model, sample_loader)

    test_model(args, model, test_loader, criterion)
