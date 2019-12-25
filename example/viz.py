import os
import matplotlib.pyplot as plt
from dataset import load_train_data


def visualize(data, target, run_name=''):
    '''
    data is of shape (B, C, H, W)
    '''
    NUM_SUBPLOTS = 27
    NUM_ROWS = 4
    fig = plt.figure(figsize=(25, 16))
    plot_num = 0
    for i in range(0, data.shape[0] - NUM_SUBPLOTS, NUM_SUBPLOTS):
        for j in range(NUM_SUBPLOTS):
            # Subplot: a x b, cth subplot
            ax = fig.add_subplot(NUM_ROWS, NUM_SUBPLOTS // NUM_ROWS + 1, j+1, label=plot_num)
            img = data[i+j].permute(1, 2, 0).squeeze()
            label = int(target[i+j])
            plt.imshow(img)
            ax.axis('off')
            ax.set_title(label)
            plot_num += 1

    if run_name:
        plt.savefig(os.path.join(run_name, 'input_data.png'))
    else:
        plt.show()


def main():
    train_loader, _ = load_train_data()
    for data, target in train_loader:
        visualize(data, target.float())


if __name__ == '__main__':
    main()
