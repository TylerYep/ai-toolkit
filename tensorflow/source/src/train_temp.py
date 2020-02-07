import sys
import random
import numpy as np
import tensorflow as tf
from tf.keras.callbacks import ModelCheckpoint, TensorBoard

# from src import util
from src.args import init_pipeline
from src.dataset  import load_train_data
# from src.metric_tracker import MetricTracker, Mode
# from src.models import BasicCNN as Model
from src.verify import verify_model
# from src.viz import visualize, visualize_trained

if 'google.colab' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def train_and_validate(model, train_images, train_labels, class_labels):
    model_name = f'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2, verbose=0,
                        callbacks[
                            ModelCheckpoint(model_name, monitor='loss'),
                            TensorBoard()
                        ]) # Save model or save weights?


# def init_metrics(args, checkpoint):
#     run_name = checkpoint.get('run_name', util.get_run_name(args))
#     return run_name, metrics


def load_model(args, checkpoint, init_params, train_images, train_labels):
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-6)
    loss = tf.keras.losses.SparseCategoricalCrossentropy() #F.nll_loss
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()
    verify_model(model, train_images, train_labels)
    # util.load_state_dict(checkpoint, model, optimizer)
    return model


def train(arg_list=None):
    args, checkpoint = init_pipeline(arg_list)
    (train_images, train_labels), (test_images, test_labels), class_labels, init_params = \
        load_train_data(args)
    model = load_model(args, checkpoint, init_params, train_images, train_labels)

    # run_name, metrics = init_metrics(args, checkpoint)
    # if args.visualize:
        # metrics.add_network(model, train_loader, device)
        # visualize(model, train_loader, class_labels, device, run_name)

    # util.set_rng_state(checkpoint)
    # start_epoch = metrics.epoch + 1
        # metrics.next_epoch()
    train_loss = train_and_validate(model, train_images, train_labels, class_labels)

        # is_best = metrics.update_best_metric(val_loss)
        # util.save_checkpoint({
        #     'model_init': init_params,
        #     'state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'rng_state': random.getstate(),
        #     'np_rng_state': np.random.get_state(),
        #     'torch_rng_state': torch.get_rng_state(),
        #     'run_name': run_name,
        #     'metric_obj': metrics.json_repr()
        # }, run_name, is_best)

    # if args.visualize:
    #     visualize_trained(model, train_loader, class_labels, device, run_name)

    # return val_loss

if __name__ == '__main__':
    train()
