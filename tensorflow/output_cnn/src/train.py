import os

import tensorflow as tf

# from src import util
from src.args import init_pipeline
from src.dataset import load_train_data
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# from src.verify import verify_model
# from src.viz import visualize, visualize_trained


def train_and_validate(
    args, model, train_images, train_labels, class_labels, save_dir="checkpoints/"
):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_name = os.path.join(save_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    history = model.fit(
        train_images,
        train_labels,
        epochs=args.epochs,
        validation_split=0.2,
        callbacks=[ModelCheckpoint(model_name, monitor="loss"), TensorBoard()],
    )  # Save model or save weights?
    return history


def load_model(checkpoint, init_params, train_images, train_labels):
    """
    Note: It is possible to bake this tf.nn.softmax in as the activation function for the
    last layer of the network. While this can make the model output more directly interpretable,
    this approach is discouraged as it's impossible to provide an exact and numerically stable
    loss calculation for all models when using a softmax output.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-6)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.summary()
    # verify_model(model, train_images, train_labels)
    # util.load_state_dict(checkpoint, model, optimizer)
    return model


def train(arg_list=None):
    args, checkpoint = init_pipeline(arg_list)
    (
        (train_images, train_labels),
        (test_images, test_labels),
        class_labels,
        init_params,
    ) = load_train_data(args)
    model = load_model(checkpoint, init_params, train_images, train_labels)
    # add_network(model, train_loader, device)
    # visualize(model, train_loader, class_labels, device, run_name)
    # util.set_rng_state(checkpoint)
    train_and_validate(args, model, train_images, train_labels, class_labels)
    # probability_model = tf.keras.Sequential([
    #     model,
    #     tf.keras.layers.Softmax()
    # ])
    # visualize_trained(model, train_loader, class_labels, device, run_name)
    # return val_loss


if __name__ == "__main__":
    train()
