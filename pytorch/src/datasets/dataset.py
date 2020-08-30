from __future__ import annotations

import sys
from typing import Any, Callable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataloader import default_collate

from src.args import Arguments


class DatasetLoader:
    def __init__(self) -> None:
        self.CLASS_LABELS: List[str] = []
        self.DATA_PATH = "/content/" if "google.colab" in sys.modules else "data/"

    def split_data(
        self,
        orig_dataset: Dataset[torch.Tensor],
        args: Arguments,
        device: torch.device,
        val_split: float,
    ) -> Tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
        collate_fn = self.get_collate_fn(device)
        generator_seed = torch.Generator().manual_seed(0)
        if args.num_examples:
            n = args.num_examples
            data_split = [n, n, len(orig_dataset) - 2 * n]
            train_set, val_set = random_split(orig_dataset, data_split, generator_seed)[:-1]
        else:
            train_size = int((1 - val_split) * len(orig_dataset))
            data_split = [train_size, len(orig_dataset) - train_size]
            train_set, val_set = random_split(orig_dataset, data_split, generator_seed)

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_fn)
        return train_loader, val_loader

    def load_train_data(
        self, args: Arguments, device: torch.device, val_split: float = 0.2
    ) -> Tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor], Tuple[Any, ...]]:
        raise NotImplementedError

    def load_test_data(self, args: Arguments, device: torch.device) -> DataLoader[torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def get_collate_fn(device: torch.device) -> Callable[[List[Any]], Any]:
        """
        for indices in batch_sampler:
            yield collate_fn([dataset[i] for i in indices])
        """

        def to_device(b: torch.Tensor) -> Any:
            return list(map(to_device, b)) if isinstance(b, (list, tuple)) else b.to(device)

        return lambda x: map(to_device, default_collate(x))
