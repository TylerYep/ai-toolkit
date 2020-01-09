import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models

import util
from args import init_pipeline
from dataset import load_train_data, INPUT_SHAPE
from models import BasicCNN as Model
from .viz_utils import jitter, rearrange


def main():
    args, _ = init_pipeline()
    train_loader, _ = load_train_data(args)
    data, target = next(iter(train_loader))
    # model = Model()#.to(device)
    model = models.resnet18(pretrained=True)
    # util.load_checkpoint(args.checkpoint, model)
    # visualize(data, target)
    # show_saliency_maps(data, target, model)
    # make_fooling_image(data, target, target[1], model)
    create_class_visualization(target[1], model)


def visualize(model, data, target, run_name=''):
    pass
    # view_input(data, target, run_name)
    # compute_activations(model, data, run_name)


def view_input(data, target, run_name=''):
    ''' Data is of shape (B, C, H, W) '''
    NUM_SUBPLOTS = 24
    NUM_ROWS = 4
    plot_num = 0
    fig = plt.figure(figsize=(25, 16))
    for i in range(0, data.shape[0] - NUM_SUBPLOTS, NUM_SUBPLOTS):
        for j in range(NUM_SUBPLOTS):
            # Subplot: a x b, cth subplot
            ax = fig.add_subplot(NUM_ROWS, NUM_SUBPLOTS // NUM_ROWS + 1, j+1, label=plot_num)
            img = rearrange(data[i+j]) #.permute(1, 2, 0).squeeze()
            # label = int(target[i+j])
            plt.imshow(img)
            ax.axis('off')
            # ax.set_title(label)
            plot_num += 1

    if run_name:
        plt.savefig(os.path.join(run_name, 'input_data.png'))
        plt.clf()
    else:
        plt.show()


def compute_activations(model, inputs, run_name=''):
    plot_num = 0
    fig = plt.figure(figsize=(25, 16))
    _, activations = model.forward_with_activations(inputs)
    NUM_SUBPLOTS = len(activations)
    for i, activation in enumerate(activations):
        activation = torch.abs(activation).mean(dim=1)[0].detach().numpy()
        activation /= activation.max()
        activation = plt.get_cmap('inferno')(activation)
        activation = np.delete(activation, 3, 2)  # deletes 4th channel created by cmap
        ax = fig.add_subplot(1, NUM_SUBPLOTS + 1, i+1, label=plot_num)
        plt.imshow(activation)
        ax.axis('off')
        plot_num += 1

    if run_name:
        plt.savefig(os.path.join(run_name, f'activation_layers.png'))
        plt.clf()
    else:
        plt.show()


def compute_saliency(inputs, run_name=''):
    # Deprecated
    saliency = inputs.grad.data
    saliency, _ = torch.max(saliency, dim=1)  # dim 1 is the channel dimension
    plt.imshow(saliency.numpy()[0], cmap=plt.cm.gray)
    plt.axis('off')
    # plt.gcf().set_size_inches(12, 5)
    if run_name:
        plt.savefig(os.path.join(run_name, 'saliency.png'))
        plt.clf()
    else:
        plt.show()


def show_saliency_maps(X, y, model, run_name=''):
    def compute_saliency_maps(X, y, model):
        """
        Compute a class saliency map using the model for images X and labels y.
        Performs a forward and backward pass through the model to compute the
        gradient of the correct class score with respect to each input image.

        Input:
        - X: Input images; Torch Tensor of shape (N, C, H, W)
        - y: Labels for X; Torch LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the saliency map.

        Returns:
        - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
        images.
        """
        model.eval()
        X.requires_grad_()

        scores = model(X)
        loss = scores.gather(1, y.view(-1, 1)).squeeze()
        loss.backward(torch.ones(scores.shape[0]))
        grad = X.grad.data
        saliency, _ = torch.max(grad.abs(), dim=1)
        return saliency

    saliency = compute_saliency_maps(X, y, model)

    # Convert the saliency map from Torch Tensor to numpy array and
    # show images and saliency maps together.
    saliency = saliency.numpy()
    N = 6
    for i in range(N):
        img = rearrange(X[i])
        plt.subplot(2, N, i + 1)
        plt.imshow(img)
        plt.axis('off')
        # plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')

    if run_name:
        plt.savefig(os.path.join(run_name, 'saliency.png'))
        plt.clf()
    else:
        plt.show()


def make_fooling_image(X, y, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224) = INPUT_SHAPE
    - target_y: An integer in the range [0, num_classes)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    idx = 2 # input img
    X_fooling = X[idx].clone().unsqueeze(dim=0)
    X_fooling.requires_grad_()

    learning_rate = 1
    for i in range(100):
        scores = model(X_fooling)
        _, index = torch.max(scores, dim=1)
        if index[0] == target_y:
            break
        loss = scores[0, target_y]
        loss.backward()
        g = X_fooling.grad.data
        dX = learning_rate * g / torch.norm(g)**2
        X_fooling.data += dX
        X_fooling.grad.data.zero_()

    X_fooling_np = X_fooling.squeeze(dim=0)

    class_names = [i for i in range(50)] # TODO
    plt.subplot(1, 4, 1)
    plt.imshow(rearrange(X[idx]))
    plt.title(class_names[y[idx]])
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(rearrange(X_fooling_np))
    plt.title(class_names[target_y])
    plt.axis('off')

    plt.subplot(1, 4, 3)
    diff = X_fooling - X[idx]
    plt.imshow(rearrange(diff))
    plt.title('Difference')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(rearrange(10 * diff))
    plt.title('Magnified difference (10x)')
    plt.axis('off')

    plt.gcf().set_size_inches(12, 5)
    plt.show()

    return X_fooling


def create_class_visualization(target_y, model, **kwargs):
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
    img = torch.randn(1, 3, 224, 224).mul_(1.0).type(dtype).requires_grad_()

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
            # class_name = class_names[target_y]
            plt.title('%s\nIteration %d / %d' % ('target', t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            plt.show()

    return img.data #deprocess(img.data.cpu())


if __name__ == '__main__':
    main()
