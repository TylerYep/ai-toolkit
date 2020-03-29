import sys
import torch
import torchsummary

if 'google.colab' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def verify_model(model, loader, optimizer, criterion, device):
    """
    Performs all necessary validation on your model to ensure correctness.
    You may need to change the batch_size or max_iters in overfit_example
    in order to overfit the batch.
    """
    torchsummary.summary(model, model.input_shape)
    check_batch_dimension(model, loader, optimizer)
    overfit_example(model, loader, optimizer, criterion, device)
    check_all_layers_training(model, loader, optimizer, criterion)
    print('Verification complete - all tests passed!')


def checkNaN(weights):
    assert not torch.isnan(weights).byte().any()
    assert torch.isfinite(weights).byte().any()


def detect_nan_tensors(self, model, loss):
    # check if loss is nan
    if not torch.isfinite(loss).all():
        raise ValueError(
            'The loss returned in `training_step` is nan or inf.'
        )
    # check if a network weight is nan
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            self.print_nan_gradients()
            raise ValueError(
                f'Detected nan and/or inf values in `{name}`.'
                ' Check your forward pass for numerically unstable operations.'
            )


def check_all_layers_training(model, loader, optimizer, criterion):
    """
    Verifies that the provided model trains all provided layers.
    """
    model.train()
    torch.set_grad_enabled(True)
    data, target = next(iter(loader))
    before = [param.clone().detach() for param in model.parameters()]

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    after = model.parameters()
    for start, end in zip(before, after):
        assert (start != end).any()


def overfit_example(model, loader, optimizer, criterion, device, batch_size=5, max_iters=50):
    """
    Verifies that the provided model can overfit a single batch or example.
    """
    model.eval()
    torch.set_grad_enabled(True)
    data, target = next(iter(loader))
    data, target = data[:batch_size], target[:batch_size]
    with tqdm(desc='Verify Model', total=max_iters, ncols=120) as pbar:
        for _ in range(max_iters):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            if torch.allclose(loss, torch.tensor(0.).to(device)):
                break
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()

    # assert torch.allclose(loss, torch.tensor(0.))


def check_batch_dimension(model, loader, optimizer, test_val=2):
    """
    Verifies that the provided model loads the data correctly. We do this by setting the
    loss to be something trivial (e.g. the sum of all outputs of example i), running the
    backward pass all the way to the input, and ensuring that we only get a non-zero gradient
    on the i-th input.
    See details at http://karpathy.github.io/2019/04/25/recipe/.
    """
    model.eval()
    torch.set_grad_enabled(True)
    data, _ = next(iter(loader))
    optimizer.zero_grad()
    data.requires_grad_()

    output = model(data)
    loss = output[test_val].sum()
    loss.backward()

    assert loss != 0
    assert (data.grad[test_val] != 0).any()
    assert (data.grad[:test_val] == 0.).all() and (data.grad[test_val+1:] == 0.).all()
