from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_data():
    BATCH_SIZE = 100
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('data', train=True, download=False, transform=norm)
    val_set = datasets.MNIST('data', train=False, transform=norm)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(val_loader))
    X_train = X_train.numpy().reshape((BATCH_SIZE, -1))
    y_train = y_train.numpy()
    X_test = X_test.numpy().reshape((BATCH_SIZE, -1))
    y_test = y_test.numpy()

    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    return X_train, y_train, X_test, y_test
