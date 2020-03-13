import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T


def save_figure(run_name, img_name):
    plt.tight_layout(True)
    if run_name:
        plt.savefig(os.path.join(run_name, img_name))
        plt.clf()
    else:
        plt.show()


def rearrange(orig_img):
    img = orig_img.clone()

    if img.requires_grad:
        img = img.detach()

    # Normalize image
    max_val = torch.max(img)
    min_val = torch.min(img)
    if img.dtype == torch.float:
        img_data_max = 1
    elif img.dtype == torch.int:
        img_data_max = 255

    if min_val < 0 or max_val > img_data_max:
        img -= img.min()
        img /= img.max()

    # Reshape
    if len(img.shape) == 4 and img.shape[0] == 1:
        img = img.squeeze(0)
    elif len(img.shape) == 2:
        img = img.unsqueeze(0)

    assert len(img.shape) == 3

    # Determine correct number of channels and permute
    if img.shape[0] == 1:
        return torch.cat([img] * 3).permute((1, 2, 0))
    if img.shape[0] == 3:
        return img.permute((1, 2, 0))
    if img.shape[-1] == 3:
        return img
    if img.shape[-1] == 1:
        return torch.cat([img] * 3)
    return img


def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X


def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def blur_image(X, sigma=1):
    from scipy.ndimage.filters import gaussian_filter1d
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X
