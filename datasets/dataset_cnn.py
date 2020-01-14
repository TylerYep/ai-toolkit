import sys
from typing import Tuple
from argparse import Namespace
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
if 'google.colab' in sys.modules:
    DATA_PATH = '/content/'
else:
    DATA_PATH = 'data/'


INPUT_SHAPE = (1, 28, 28)


# def load_train_data(args: Namespace) -> Tuple[DataLoader, DataLoader]:
#     norm = get_transforms()
#     train_set = datasets.ImageFolder('data/tiny-imagenet-200', transform=norm)
#     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
#     return train_loader, None, {}


def load_train_data(args: Namespace) -> Tuple[DataLoader, DataLoader]:
    norm = get_transforms()
    train_set = datasets.FashionMNIST(DATA_PATH, train=True, download=True, transform=norm)
    val_set = datasets.FashionMNIST(DATA_PATH, train=False, transform=norm)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.test_batch_size)
    return train_loader, val_loader, {}


def load_test_data(args: Namespace) -> DataLoader:
    norm = get_transforms()
    test_set = datasets.FashionMNIST(DATA_PATH, train=False, transform=norm)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size)
    return test_loader


def get_transforms():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # norm = transforms.Compose([transforms.Grayscale(num_output_channels=3),
    #                            transforms.Resize(224),
    #                            transforms.ToTensor(),
    #                            transforms.Normalize((0.1307,), (0.3081,))])


class MyDataset(Dataset):
    ''' Dataset for training a model on a dataset. '''
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.label = pd.read_csv(data_path)
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
