import sys
import warnings
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo

from ai_toolkit.args import Arguments

if "google.colab" in sys.modules:
    from tqdm import tqdm_notebook as tqdm  # type: ignore[import]
else:
    from tqdm import tqdm


def verify_model(
    args: Arguments,
    model: nn.Module,
    loader: Iterator[Any],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    """
    Performs all necessary validation on your model to ensure correctness.
    You may need to change the batch_size or max_iters in overfit_example
    in order to overfit the batch.
    """
    if not args.no_verify:
        model_summary(args, model, loader)
        check_batch_dimension(model, loader, optimizer)
        overfit_example(model, loader, optimizer, criterion, device, args.batch_dim)
        check_all_layers_training(model, loader, optimizer, criterion)
        detect_NaN_tensors(model)
        print("Verification complete - all tests passed!")


def model_summary(args: Arguments, model: nn.Module, loader: Iterator[Any]) -> None:
    """
    Prints out model using torchsummary.
    """
    data, _ = next(loader)
    torchinfo.summary(model, input_data=data, batch_dim=args.batch_dim)


def check_batch_dimension(
    model: nn.Module,
    loader: Iterator[Any],
    optimizer: optim.Optimizer,
    test_val: int = 2,
) -> None:
    """
    Verifies that the provided model loads the data correctly. We do this by
    setting the loss to be something trivial (e.g. the sum of all outputs
    of example i), running the backward pass all the way to the input,
    and ensuring that we only get a non-zero gradient on the i-th input.
    See details at http://karpathy.github.io/2019/04/25/recipe/.
    """
    model.eval()
    data, _ = next(loader)
    optimizer.zero_grad()

    if not isinstance(data, (list, tuple)):
        data.requires_grad_()
        output = model(data)
        loss = output[test_val].sum()
        loss.backward()

        if loss == 0:
            raise RuntimeError("Loss should be greater than zero.")
        if (data.grad[test_val] == 0).all():
            raise RuntimeError("Grad of test input is not nonzero.")
        if (data.grad[:test_val] != 0).any() and (
            (data.grad[test_val + 1 :] != 0).any()
        ):
            raise RuntimeError(
                "Batch contains nonzero gradients, when they should all be zero."
            )


def overfit_example(
    model: nn.Module,
    loader: Iterator[Any],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_dim: int = 0,
    batch_size: int = 2,
    max_iters: int = 50,
) -> None:
    """
    Verifies that the provided model can overfit a single batch or example.
    """

    def batch_slice(input_data: Any, batch_size: int, batch_dim: int) -> Any:
        if isinstance(input_data, (list, tuple)):
            return [batch_slice(data, batch_size, batch_dim) for data in input_data]
        if input_data.ndim == 1:
            return input_data[:batch_size]
        none_slice = (slice(None),)
        batch_dim_slice = (
            none_slice * batch_dim
            + (slice(batch_size),)
            + none_slice * (input_data.ndim - batch_dim - 1)
        )
        return input_data[batch_dim_slice]

    model.train()
    data, target = next(loader)
    data = batch_slice(data, batch_size, batch_dim)
    target = batch_slice(target, batch_size, batch_dim)

    with tqdm(desc="Verify Model", total=max_iters, ncols=120) as pbar:
        for _ in range(max_iters):
            optimizer.zero_grad()
            output = model(*data) if isinstance(data, (list, tuple)) else model(data)
            loss = criterion(output, target)
            if torch.allclose(loss, torch.tensor(0.0).to(device)):
                break
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"Loss": loss.item()})
            pbar.update()

    if not torch.allclose(loss, torch.tensor(0.0).to(device)):
        warnings.warn(
            f"\nOverfit Loss is not sufficiently close to 0: {loss}\n"
            "This may indicate an error with your model.",
            RuntimeWarning,
        )


def detect_NaN_tensors(model: nn.Module) -> None:
    """
    Verifies that the provided model does not have any exploding gradients.
    """
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(
                f"Detected NaN and/or inf values in model weights: {name}. "
                "Check your forward pass for numerically unstable operations."
            )


def check_all_layers_training(
    model: nn.Module,
    loader: Iterator[Any],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> None:
    """
    Verifies that the provided model trains all provided layers.
    """
    model.train()
    data, target = next(loader)
    before = [param.clone().detach() for param in model.parameters()]

    optimizer.zero_grad()
    output = model(*data) if isinstance(data, (list, tuple)) else model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    after = model.parameters()
    for start, end in zip(before, after):
        if (start == end).all():
            raise RuntimeError(
                "Detected some layers that are not training. Did you freeze "
                "some layers or forget to set the model to train mode?"
            )
