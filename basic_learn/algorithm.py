import pandas as pd
import numpy as np
import itertools
import sklearn
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

import os, sys
sys.path.append('ml-toolkit')
import util

def load_alg(name):
    path = 'data/ml/results/' + name +'.pkl'
    if os.path.isfile(path):
        return util.load_pkl(path)
    model_name = name.split('-')[0]
    return Algorithm(name, util.model_dict[model_name])

class Algorithm:
    def __init__(self, name, model):
        """
        Args:
            name: name of model
            model: uninstantiated sklearn model
        """
        self.name = name
        self.model = model
        self.results = pd.DataFrame(columns=util.results_headers)

    def get_fname(self):
        # Returns the file path to save
        fname = self.name
        fname += '.pkl'
        return os.path.join('data/ml/results', fname)

    def save(self):
        util.save_pkl(self.get_fname(), self)

    def predict(self, x):
        return self.clf.predict(x)

    def train(self, x, y):
        self.clf.fit(x, y)
        preds = self.predict(x)
        return mean_squared_error(y, preds)
        # return util.get_acc(y, preds)

    def eval(self, x, y):
        predictions = self.predict(x)
        # test_error = util.get_acc(y, predictions)
        test_error = mean_squared_error(y, predictions)
        return test_error


    def run(self, data, features, clf_options={}):
        """
        Arguments
            data: DataFeatures object
            features: list of features
            clf_options: dictionary of sklearn classifier options
        """
        self.clf = self.model(**clf_options)
        X = data.get_joint_matrix(util.features)
        train_x = X[data.train_indices]
        train_y = data.labels[data.train_indices]
        val_x = X[data.val_indices]
        val_y = data.labels[data.val_indices]
        test_x = X[data.test_indices]
        test_y = data.labels[data.test_indices]

        train_acc = self.train(train_x, train_y)
        dev_acc = self.eval(val_x, val_y)
        test_acc = self.eval(test_x, test_y)

        # Add a row to results
        row = (self.name, str(clf_options), train_acc, dev_acc, test_acc)
        self.results.loc[len(self.results)] = row
        self.save()

    def search(self, data, param_dist, features):
        X = data.get_joint_matrix(features)
        y = data.labels
        r_search = RandomizedSearchCV(self.model(), param_dist, n_iter=20)
        r_search.fit(X, y)
        self.r = r_search

    def to_csv(self):
        self.results.to_csv(os.path.join('data/ml/results', self.name + '.csv'), index=False)

    def conf_matrix(self, x, y):
        def plot_confusion_matrix(cm, classes, normalize=False,  title='Confusion matrix', cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=0)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
        predictions = self.predict(x)
        cnf_matrix = confusion_matrix(y, predictions)
        np.set_printoptions(precision=2)
        plot_confusion_matrix(cnf_matrix, classes=['level '+ str(i) for i in [2, 3, 4]], \
                                normalize=True, title='Confusion Matrix for Logistic Regression')
        #plt.imshow(conf)
        plt.show()