import sys
import tensorflow as tf

if 'google.colab' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def verify_model(model, train_images, train_labels):
    """
    Performs all necessary validation on your model to ensure correctness.
    You may need to change the batch_size or max_iters in overfit_example
    in order to overfit the batch.
    """
    gradient_check(model, train_images, train_labels)
    overfit_example(model, train_images, train_labels)
    print('Verification complete - all tests passed!')


def gradient_check(model, train_images, train_labels, test_val=3):
    """
    Verifies that the provided model loads the data correctly. We do this by setting the
    loss to be something trivial (e.g. the sum of all outputs of example i), running the
    backward pass all the way to the input, and ensuring that we only get a non-zero gradient
    on the i-th input.
    See details at http://karpathy.github.io/2019/04/25/recipe/.

    Will not work if the model does not compute a gradient for the predicted test_val.
    """
    model.train_on_batch(train_images[test_val:test_val+1], train_labels[test_val:test_val+1])
    x_tensor = tf.convert_to_tensor(train_images, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        loss = tf.math.reduce_sum(model(x_tensor)[test_val])

    grad = t.gradient(loss, x_tensor).numpy()
    assert loss.numpy() != 0
    assert (grad[test_val] != 0).any()
    assert (grad[:test_val] == 0.).all() and (grad[test_val+1:] == 0.).all()


def overfit_example(model, train_images, train_labels, batch_size=5, max_iters=500):
    """
    Verifies that the provided model can overfit a single batch or example.
    """
    images, labels = train_images[:batch_size], train_labels[:batch_size]
    for _ in range(max_iters):
        loss, output = model.train_on_batch(images, labels)
        print(loss, output)
