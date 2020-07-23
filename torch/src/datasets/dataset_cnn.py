import sys

import pandas as pd
from torchvision import datasets, transforms

import torch
from src.datasets.dataset import DatasetLoader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset


class DatasetCNN(DatasetLoader):
    def __init__(self):
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

    def load_train_data(self, args, device, val_split=0.2):
        orig_dataset = datasets.FashionMNIST(
            self.DATA_PATH, train=True, download=True, transform=self.get_transforms()
        )
        train_loader, val_loader = self.split_data(orig_dataset, args, device, val_split)
        init_params = [torch.Size((1, 28, 28))]
        return train_loader, val_loader, init_params

    def load_test_data(self, args, device):
        collate_fn = self.get_collate_fn(device)
        test_set = datasets.FashionMNIST(
            self.DATA_PATH, train=False, transform=self.get_transforms()
        )
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=collate_fn)
        return test_loader

    @staticmethod
    def get_collate_fn(device):
        """
        for indices in batch_sampler:
            yield collate_fn([dataset[i] for i in indices])
        """

        def to_device(b):
            return list(map(to_device, b)) if isinstance(b, (list, tuple)) else b.to(device)

        return lambda x: map(to_device, default_collate(x))

    @staticmethod
    def get_transforms(img_dim=None):
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
