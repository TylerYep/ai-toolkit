from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms


def load_data(args):
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('data', train=True, download=True, transform=norm)
    test_set = datasets.MNIST('data', train=False, transform=norm)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size)
    return train_loader, test_loader, test_loader


class MyDataset(Dataset):
    """ Dataset for training a model on a dataset. """
    def __init__(self):
        super().__init__()
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return data[index]
