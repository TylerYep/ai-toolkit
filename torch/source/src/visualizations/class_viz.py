import random
import matplotlib.pyplot as plt
import torch

from .viz_utils import jitter, rearrange, save_figure

def create_class_visualization(model, data, class_labels, target_y, run_name, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.

    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    - dtype: Torch datatype to use for computations

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    dtype = torch.FloatTensor
    model.type(dtype)
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)

    # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.
    INPUT_SHAPE = data[0].unsqueeze(0).shape
    img = torch.randn(INPUT_SHAPE).mul_(1.0).type(dtype).requires_grad_()

    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
        img.data.copy_(jitter(img.data, ox, oy))

        scores = model(img)
        target_score = scores[0, target_y]
        target_score.backward()
        g = img.grad.data - 2 * l2_reg * img.data
        img.data += learning_rate * g
        img.grad.data.zero_()

        # Undo the random jitter
        img.data.copy_(jitter(img.data, -ox, -oy))

        # As regularizer, clamp and periodically blur the image
        # for c in range(3):
        #     lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
        #     hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
        #     img.data[:, c].clamp_(min=lo, max=hi)
        # if t % blur_every == 0:
        #     blur_image(img.data, sigma=0.5)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(rearrange(img.data))
            plt.title(f'{class_labels[target_y]}\nIteration {t + 1} / {num_iterations}')
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            # plt.show()

    save_figure(run_name, 'class_viz.png')
    return img.data  # deprocess(img.data.cpu())
