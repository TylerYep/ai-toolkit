import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
if 'google.colab' in sys.modules:
    DATA_PATH = '/content/'
else:
    DATA_PATH = 'data/'


INPUT_SHAPE = (1, 28, 28)
# CLASS_LABELS = [str(i) for i in range(10)]
CLASS_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# def load_train_data(args):
#     norm = get_transforms()
#     train_set = datasets.ImageFolder('data/tiny-imagenet-200', transform=norm)
#     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
#     return train_loader, None, [0 for _ in range(10000)], {}


def get_collate_fn(device):
    return lambda x: map(lambda b: b.to(device), default_collate(x))


def load_train_data(args, device):
    norm = get_transforms()
    collate_fn = get_collate_fn(device)
    train_set = datasets.FashionMNIST(DATA_PATH, train=True, download=True, transform=norm)
    val_set = datasets.FashionMNIST(DATA_PATH, train=False, transform=norm)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_set,
                            batch_size=args.test_batch_size,
                            collate_fn=collate_fn)
    return train_loader, val_loader, {}


def load_test_data(args, device):
    norm = get_transforms()
    collate_fn = get_collate_fn(device)
    test_set = datasets.FashionMNIST(DATA_PATH, train=False, transform=norm)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             collate_fn=collate_fn)
    return test_loader


def get_transforms():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])
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
