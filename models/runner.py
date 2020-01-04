import math
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_data, preprocess
from classifiers import KNearestNeighbor, LinearSVM, Softmax, TwoLayerNet
# from k_fold import k_fold_validation


def main():
    np.random.seed(0)
    X_train, y_train, X_test, y_test = load_data()
    # X_train, X_test = preprocess(X_train, X_test)
    # classifiers = [KNearestNeighbor, LinearSVM, Softmax, TwoLayerNet]
    # clf_dict = {clf.__name__: clf for clf in classifiers}

    # ###
    # clf_name = 'LinearSVM'
    # kwargs = {

    # }
    # ###

    # clf = clf_dict[clf_name](**kwargs)
    # loss_hist = clf.fit(X_train, y_train)
    # y_test_pred = clf.predict(X_test)

    # num_correct = np.sum(y_test_pred == y_test)
    # num_test = y_test.shape[0]
    # accuracy = float(num_correct) / num_test
    # print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
    # k_fold_validation(X_train, y_train)
    tune_svm(X_train, y_train, X_test, y_test)


def plot_loss_curve(loss_hist):
    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.title('Loss Curve')
    plt.show()


def visualize_weights(clf, classes=None):
    w = clf.W[:-1, :] # strip out the bias
    w = w.reshape(28, 28, 1, 10) # (h, w, c, b)
    w_min, w_max = np.min(w), np.max(w)
    # classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.subplot(2, 5, i + 1)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        if classes is not None:
            plt.title(classes[i])

    plt.show()


def tune_svm(X_train, y_train, X_val, y_val):
    learning_rates = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    regularization_strengths = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6]

    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the fraction
    # of data points that are correctly classified.
    results = {}
    best_val = -1   # The highest validation accuracy that we have seen so far.
    best_svm = None # The LinearSVM object that achieved the highest validation rate.

    for lr in learning_rates:
        for reg in regularization_strengths:
            print(lr, reg)
            svm = LinearSVM()
            svm.fit(X_train, y_train, lr, reg, num_iters=100, verbose=True)
            y_train_pred = svm.predict(X_train)
            y_val_pred = svm.predict(X_val)
            train_acc = np.mean(y_train == y_train_pred)
            val_acc = np.mean(y_val_pred == y_val)
            results[(lr, reg)] = (train_acc, val_acc)
            if best_val < val_acc:
                best_val = val_acc
                best_svm = svm

    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')

    # plot validation accuracy
    colors = [results[x][1] for x in results] # default size of markers is 20
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.show()


if __name__ == '__main__':
    main()
