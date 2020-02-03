import sys
import pandas as pd
import tensorflow as tf
if 'google.colab' in sys.modules:
    DATA_PATH = '/content/'
else:
    DATA_PATH = 'data/'


INPUT_SHAPE = (1, 28, 28)


def load_train_data(args):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels), class_names, {}


# def load_test_data(args):
#     norm = get_transforms()
#     test_set = datasets.FashionMNIST(DATA_PATH, train=False, transform=norm)
#     test_loader = DataLoader(test_set, batch_size=args.test_batch_size)
#     return test_loader
