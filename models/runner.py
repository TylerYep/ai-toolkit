import numpy as np

from dataset import load_data
from classifiers import KNearestNeighbor, LinearSVM, Softmax, TwoLayerNet
from k_fold import k_fold_validation

def main(use_sklearn=False):
    X_train, y_train, X_test, y_test = load_data()
    # k_fold_validation(X_train, y_train)

    clf = LinearSVM()
    # clf = KNearestNeighbor(k=5)
    # clf = Softmax()
    # clf = TwoLayerNet()

    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    num_correct = np.sum(y_test_pred == y_test)
    num_test = y_test.shape[0]
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


if __name__ == '__main__':
    main()
