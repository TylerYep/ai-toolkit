import torch
import torch.optim as optim

import util
from args import init_pipeline
from dataset import load_train_data, INPUT_SHAPE
from models import BasicCNN as Model


def main():
    args, device, _ = init_pipeline()
    train_loader, _, class_labels, init_params = load_train_data(args)
    model = Model(*init_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    verify_model(model, train_loader, optimizer, device, test_val=2)
    print('Verification complete - all tests passed!')


def verify_model(model, loader, optimizer, device, test_val=2):
    """
    Verifies that the provided model loads the data correctly. We do this by setting the
    loss to be something trivial (e.g. the sum of all outputs of example i), running the
    backward pass all the way to the input, and ensuring that we only get a non-zero gradient
    on the i-th input.
    See details at http://karpathy.github.io/2019/04/25/recipe/.
    """
    model.eval()
    torch.set_grad_enabled(True)
    data, target = util.get_data_example(loader, device)
    optimizer.zero_grad()
    data.requires_grad_()

    output = model(data)
    loss = output[test_val].sum()
    loss.backward()

    assert loss.data != 0
    assert (data.grad[test_val] != 0).any()
    assert (data.grad[:test_val] == 0.).all() and (data.grad[test_val+1:] == 0.).all()


if __name__ == '__main__':
    main()
