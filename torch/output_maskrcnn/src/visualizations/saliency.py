import matplotlib.pyplot as plt

import torch

from .viz_utils import rearrange, save_figure


def compute_saliency(inputs, run_name):
    # Deprecated
    saliency = inputs.grad.data
    saliency, _ = torch.max(saliency, dim=1)  # dim 1 is the channel dimension
    plt.imshow(saliency.numpy()[0], cmap=plt.cm.gray)
    plt.axis("off")
    save_figure(run_name, "saliency.png")


def show_saliency_maps(model, X, y, class_labels, run_name):
    def compute_saliency_maps(model, X, y):
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

    saliency = compute_saliency_maps(model, X, y)

    # Convert the saliency map from Torch Tensor to numpy array and
    # show images and saliency maps together.
    saliency = saliency.numpy()
    N = 6
    for i in range(N):
        img = rearrange(X[i])
        plt.subplot(2, N, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(class_labels[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis("off")

    save_figure(run_name, "saliency.png")
