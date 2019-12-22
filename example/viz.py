import matplotlib.pyplot as plt
from dataset import load_data


def visualize(data, target):
    '''
    data is of shape (B, C, H, W)
    '''
    NUM_SUBPLOTS = 27
    NUM_ROWS = 4
    fig = plt.figure(figsize=(25, 16))
    for i in range(0, data.shape[0] - NUM_SUBPLOTS, NUM_SUBPLOTS):
        for j in range(NUM_SUBPLOTS):
            # Subplot: a x b, cth subplot
            ax = fig.add_subplot(NUM_ROWS, NUM_SUBPLOTS // NUM_ROWS + 1, j+1)
            img = data[i+j].permute(1, 2, 0).squeeze()
            label = int(target[i+j])
            plt.imshow(img)
            ax.axis('off')
            ax.set_title(label)
    plt.show()


def main():
    train_loader, _, _ = load_data()
    for i, (data, target) in enumerate(train_loader):
        visualize(data, target.float())


if __name__ == '__main__':
    main()