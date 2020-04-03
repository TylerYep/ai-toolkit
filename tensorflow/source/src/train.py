import os
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from src import util
from src.args import init_pipeline
from src.dataset import load_train_data
from src.verify import verify_model
# from src.viz import visualize, visualize_trained

if 'google.colab' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


save_dir = 'checkpoints/'


def train_and_validate(model, train_images, train_labels, class_labels):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_name = os.path.join(save_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    history = model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_split=0.2,
        callbacks=[
            ModelCheckpoint(model_name, monitor='loss'),
            TensorBoard()
        ]) # Save model or save weights?


def load_model(args, checkpoint, init_params, train_images, train_labels):
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-6)
    loss = tf.keras.losses.SparseCategoricalCrossentropy() # from_logits=True
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()
    # verify_model(model, train_images, train_labels)
    # util.load_state_dict(checkpoint, model, optimizer)
    return model


def train(arg_list=None):
    args, checkpoint = init_pipeline(arg_list)
    (train_images, train_labels), (test_images, test_labels), class_labels, init_params = \
        load_train_data(args)
    model = load_model(args, checkpoint, init_params, train_images, train_labels)
    # add_network(model, train_loader, device)
    # visualize(model, train_loader, class_labels, device, run_name)
    # util.set_rng_state(checkpoint)
    train_and_validate(model, train_images, train_labels, class_labels)
    # visualize_trained(model, train_loader, class_labels, device, run_name)
    # return val_loss

if __name__ == '__main__':
    train()
