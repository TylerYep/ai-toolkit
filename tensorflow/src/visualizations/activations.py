from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


def compute_activations(model: Any, train_images: Any, train_labels: Any) -> None:
    output, activations = model.call_with_activations(train_images)
    NUM_EXAMPLES = 2
    NUM_SUBPLOTS = NUM_EXAMPLES * len(activations)
    _, axs = plt.subplots(NUM_SUBPLOTS // NUM_EXAMPLES, NUM_EXAMPLES)
    for i in range(NUM_EXAMPLES):
        for j, activ in enumerate(activations):
            activation = tf.math.reduce_mean(tf.abs(activ), axis=3)[i]
            activation = activation.numpy()
            activation /= activation.max()
            activation = plt.get_cmap("inferno")(activation)
            activation = np.delete(
                activation, 3, 2
            )  # deletes 4th channel created by cmap

            ax = axs[j, i]
            ax.imshow(activation)
            ax.axis("off")


# train_img, train_label = next(iter(train_dataset))
# compute_activations(model, train_img, train_label)
