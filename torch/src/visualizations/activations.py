import matplotlib.pyplot as plt
import numpy as np

import torch

from .viz_utils import save_figure


def compute_activations(model, data, target, class_labels, run_name):
    model.eval()
    _, activations = model.forward_with_activations(data)
    NUM_EXAMPLES = 4
    NUM_SUBPLOTS = NUM_EXAMPLES * len(activations)
    _, axs = plt.subplots(NUM_SUBPLOTS // NUM_EXAMPLES, NUM_EXAMPLES)
    for i in range(NUM_EXAMPLES):
        for j, activ in enumerate(activations):
            activation = torch.abs(activ).mean(dim=1)[i]
            activation = activation.detach().cpu().numpy()
            activation /= activation.max()
            activation = plt.get_cmap("inferno")(activation)
            activation = np.delete(activation, 3, 2)  # deletes 4th channel created by cmap

            ax = axs[j, i]
            ax.imshow(activation)
            ax.axis("off")
            ax.set_title(class_labels[target[i]] if j == 0 else "")

    save_figure(run_name, "activation_layers.png")


# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.cl1 = nn.Linear(25, 60)
#         self.cl2 = nn.Linear(60, 16)
#         self.fc1 = nn.Linear(16, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = F.relu(self.cl1(x))
#         x = F.relu(self.cl2(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.log_softmax(self.fc3(x), dim=1)
#         return x


# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook


# model = MyModel()
# model.fc2.register_forward_hook(get_activation('fc2'))
# x = torch.randn(1, 25)
# output = model(x)
# print(activation['fc2'])
