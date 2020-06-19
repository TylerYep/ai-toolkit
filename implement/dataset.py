import numpy as np
from torchvision import datasets, transforms

from torch.utils.data import DataLoader


def load_data():
    BATCH_SIZE = 100
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST("data", train=True, download=False, transform=norm)
    val_set = datasets.MNIST("data", train=False, transform=norm)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(val_loader))
    X_train = X_train.numpy().reshape((BATCH_SIZE, -1))
    y_train = y_train.numpy()
    X_test = X_test.numpy().reshape((BATCH_SIZE, -1))
    y_test = y_test.numpy()

    print("Training data shape: ", X_train.shape)
    print("Training labels shape: ", y_train.shape)
    print("Test data shape: ", X_test.shape)
    print("Test labels shape: ", y_test.shape)
    return X_train, y_train, X_test, y_test


def preprocess(X_train, X_test):
    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis=0)

    # second: subtract the mean image from train and test data
    X_train -= mean_image
    X_test -= mean_image

    # third: append the bias dimension of ones (i.e. bias trick) so that our SVM
    # only has to worry about optimizing a single weight matrix W.
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    return X_train, X_test
