import tensorflow as tf

INPUT_SHAPE = (1, 28, 28)


def load_train_data():
    (
        (train_images, train_labels),
        (test_images, test_labels),
    ) = tf.keras.datasets.fashion_mnist.load_data()
    class_names = [
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

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels), class_names, {}


# def load_test_data(args):
#     test_set = datasets.FashionMNIST(DATA_PATH, train=False, transform=norm)
#     return test_loader
