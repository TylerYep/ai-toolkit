import sys
import warnings
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
    detect_NaN_tensors(model)
    print('Verification complete - all tests passed!')


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

    assert loss != 0, "Loss is not exactly zero."
    assert (data.grad[test_val] != 0).any(), "The gradient of the test input is not nonzero."
    assert (data.grad[:test_val] == 0.).all() and (data.grad[test_val+1:] == 0.).all(), \
        "All other inputs in the batch are not zero."


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

    if not torch.allclose(loss, torch.tensor(0.).to(device)):
        warnings.warn(f"Overfit Loss is not sufficiently close to 0: {loss}"
                      f"This may indicate an error with your model.", RuntimeWarning)


def checkNaN(weights):
    assert not torch.isnan(weights).byte().any()
    assert torch.isfinite(weights).byte().any()


def detect_NaN_tensors(model):
    """
    Verifies that the provided model does not have any exploding gradients.
    """
    # assert torch.isfinite(loss).all(), 'The loss returned in `training_step` is NaN or inf.'
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), \
            (f'Detected NaN and/or inf values in model weights: {name}. '
             f'Check your forward pass for numerically unstable operations.')


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
        assert (start != end).any(), \
            ('Detected some layers that are not training. Did you freeze '
             'some layers or forget to set the model to train mode?')
