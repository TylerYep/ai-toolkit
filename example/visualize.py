import matplotlib.pyplot as plt
from dataset import load_data
import numpy as np

def main():
    train_loader, test_loader = load_data()
    for i, (input, targ) in enumerate(train_loader):
        target = targ.float()
        visualize(input, target)

def visualize(input, target):
    fig = plt.figure(figsize=(25, 16))
    for i in range(input.shape[0]):
        ax = fig.add_subplot(3, 2, i+1)
        plt.imshow(input[i].permute(1, 2, 0))
        ax.set_title(int(target[i]))
    plt.show()

if __name__ == '__main__':
    main()