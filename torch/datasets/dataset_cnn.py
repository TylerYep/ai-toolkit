import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
if 'google.colab' in sys.modules:
    DATA_PATH = '/content/'
else:
    DATA_PATH = 'data/'


INPUT_SHAPE = (1, 28, 28)
CLASS_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_collate_fn(device):
    return lambda x: map(lambda b: b.to(device), default_collate(x))


def load_train_data(args, device, num_examples=None, val_split=0.2):
    norm = get_transforms()
    collate_fn = get_collate_fn(device)
    orig_dataset = datasets.FashionMNIST(DATA_PATH, train=True, download=True, transform=norm)
    if num_examples:
        data_split = [num_examples, num_examples, len(orig_dataset) - 2 * num_examples]
        train_set, val_set = random_split(orig_dataset, data_split)[:-1]
    else:
        data_split = [int(part * len(orig_dataset)) for part in (1 - val_split, val_split)]
        train_set, val_set = random_split(orig_dataset, data_split)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            collate_fn=collate_fn)
    return train_loader, val_loader, []


def load_test_data(args, device):
    norm = get_transforms()
    collate_fn = get_collate_fn(device)
    test_set = datasets.FashionMNIST(DATA_PATH, train=False, transform=norm)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             collate_fn=collate_fn)
    return test_loader


def get_transforms(img_dim=None):
    # return transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.ToPILImage(),
    #     transforms.Resize((img_dim, img_dim)),
    #     transforms.ToTensor(),
    # ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
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
