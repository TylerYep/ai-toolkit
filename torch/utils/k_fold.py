import numpy as np
import matplotlib.pyplot as plt

from classifiers import KNearestNeighbor


def k_fold_validation(X_train, y_train, num_folds=5):
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    # Split up the training data into folds. After splitting, X_train_folds and
    # y_train_folds should each be lists of length num_folds, where
    # y_train_folds[i] is the label vector for the points in X_train_folds[i].
    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.
    params_to_accuracies = {}

    # Perform k-fold cross validation to find the best value of k. For each
    # possible value of k, run the k-nearest-neighbor algorithm num_folds times,
    # where in each case you use all but one of the folds as training data and the
    # last fold as a validation set. Store the accuracies for all fold and all
    # values of k in the k_to_accuracies dictionary.
    for k in k_choices:
        params_to_accuracies[k] = []
        for i in range(num_folds):
            classifier = KNearestNeighbor(k)
            classifier.fit(np.concatenate(X_train_folds[:i] + X_train_folds[i+1:]),
                           np.concatenate(y_train_folds[:i] + y_train_folds[i+1:]))
            prediction = classifier.predict(X_train_folds[i])
            num_correct = np.sum(prediction == y_train_folds[i])
            accuracy = float(num_correct) / len(y_train_folds[i])
            params_to_accuracies[k].append(accuracy)

    # plot the raw observations
    for k, accuracies in params_to_accuracies.items():
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(params_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(params_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

    return max(params_to_accuracies, key=lambda k: params_to_accuracies[k])
