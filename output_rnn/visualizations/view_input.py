import matplotlib.pyplot as plt
from .viz_utils import rearrange, save_figure


def view_input(data, target, class_labels, run_name):
    ''' Data is of shape (B, C, H, W) '''
    NUM_EXAMPLES = 15
    NUM_ROWS = 4
    _, axs = plt.subplots(NUM_ROWS, NUM_EXAMPLES // NUM_ROWS + 1)
    data, target = data.cpu(), target.cpu()
    for i, ax in enumerate(axs.flat):
        img = rearrange(data[i])
        label = class_labels[target[i]]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(label)

    save_figure(run_name, 'input_data.png')
