import matplotlib.pyplot as plt
import torch

from .viz_utils import rearrange, save_figure


def make_fooling_image(model, X, y, class_labels, target_y, run_name):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (3, 224, 224) = INPUT_SHAPE
    - target_y: An integer in the range [0, num_classes)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    assert class_labels[y] != class_labels[target_y]
    X_fooling = X.clone().unsqueeze(dim=0)
    X_fooling.requires_grad_()

    learning_rate = 1
    for _ in range(100):
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

    _, axs = plt.subplots(1, 4)
    ax = axs.flat[0]

    ax.imshow(rearrange(X))
    ax.set_title(class_labels[y])
    ax.axis('off')

    ax = axs.flat[1]
    ax.imshow(rearrange(X_fooling_np))
    ax.set_title(class_labels[target_y])
    ax.axis('off')

    ax = axs.flat[2]
    diff = X_fooling - X
    ax.imshow(rearrange(diff))
    ax.set_title('Difference')
    ax.axis('off')

    ax = axs.flat[3]
    ax.imshow(rearrange(10 * diff))
    ax.set_title('Magnified difference (10x)')
    ax.axis('off')

    plt.gcf().set_size_inches(12, 5)
    save_figure(run_name, 'fooling.png')
