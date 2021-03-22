from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.dataloader import default_collate

from src.args import Arguments

TensorDataLoader = DataLoader[Tuple[torch.Tensor, ...]]


class DatasetLoader:
    def __init__(self) -> None:
        self.CLASS_LABELS: list[str] = []
        self.DATA_PATH = Path("/content/" if "google.colab" in sys.modules else "data/")

    @staticmethod
    def get_collate_fn(device: torch.device) -> Callable[[list[Any]], Any]:
        """
        for indices in batch_sampler:
            yield collate_fn([dataset[i] for i in indices])
        """

        def to_device(b: torch.Tensor) -> Any:
            return (
                list(map(to_device, b))
                if isinstance(b, (list, tuple))
                else b.to(device)
            )

        return lambda x: map(to_device, default_collate(x))

    def split_data(
        self,
        orig_dataset: TensorDataset,
        args: Arguments,
        device: torch.device,
        val_split: float,
    ) -> tuple[TensorDataLoader, TensorDataLoader]:
        collate_fn = self.get_collate_fn(device)
        generator_seed = torch.Generator().manual_seed(0)
        orig_len = len(orig_dataset)
        if args.num_examples:
            n = args.num_examples
            data_split = [n, n, orig_len - 2 * n]
            train_set, val_set = random_split(orig_dataset, data_split, generator_seed)[
                :-1
            ]
        else:
            train_size = int((1 - val_split) * orig_len)
            data_split = [train_size, orig_len - train_size]
            train_set, val_set = random_split(orig_dataset, data_split, generator_seed)

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            num_workers=args.num_workers,
        )
        return train_loader, val_loader

    def load_train_data(
        self, args: Arguments, device: torch.device, val_split: float = 0.2
    ) -> tuple[TensorDataLoader, TensorDataLoader, tuple[Any, ...]]:
        raise NotImplementedError

    def load_test_data(self, args: Arguments, device: torch.device) -> TensorDataLoader:
        raise NotImplementedError
