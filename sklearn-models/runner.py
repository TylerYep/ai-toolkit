from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, \
                                Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import shuffle
from sklearn import metrics

from dataset import load_data


def main():
    X_train, y_train, X_test, y_test = load_data()
    classifiers = [
        KNeighborsClassifier,
        DecisionTreeClassifier,
        LogisticRegression,
        LinearRegression
    ]
    clf_dict = {clf.__name__: clf for clf in classifiers}

    ###
    clf_name = 'LinearSVM'
    kwargs = {}
    ###

    clf = clf_dict[clf_name](**kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # print("R2 Score:", metrics.r2_score(y_test, y_pred))


if __name__ == '__main__':
    main()
