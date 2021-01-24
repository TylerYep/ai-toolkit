from typing import Any

import tensorflow as tf

INPUT_SHAPE = (1, 28, 28)


class DatasetCNN:
    @staticmethod
    def load_train_data() -> Any:
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

        train_images = train_images / 255
        test_images = test_images / 255
        return (train_images, train_labels), (test_images, test_labels), class_names, {}

    @staticmethod
    def load_test_data():
        return DatasetCNN.load_train_data()
