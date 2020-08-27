from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.args import Arguments
from src.datasets.dataset import DatasetLoader


class DatasetCNN(DatasetLoader):
    def __init__(self) -> None:
        super().__init__()
        self.CLASS_LABELS = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

    def load_train_data(
        self, args: Arguments, device: torch.device, val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, Tuple[Any, ...]]:
        orig_dataset = datasets.FashionMNIST(
            self.DATA_PATH, train=True, download=True, transform=self.get_transforms()
        )
        train_loader, val_loader = self.split_data(orig_dataset, args, device, val_split)
        init_params = (torch.Size((1, 28, 28)),)
        return train_loader, val_loader, init_params

    def load_test_data(self, args: Arguments, device: torch.device) -> DataLoader:
        collate_fn = self.get_collate_fn(device)
        test_set = datasets.FashionMNIST(
            self.DATA_PATH, train=False, transform=self.get_transforms()
        )
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=collate_fn)
        return test_loader

    @staticmethod
    def get_transforms(img_dim=None):
        del img_dim
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
