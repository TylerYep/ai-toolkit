from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from dataset import load_data


def main():
    X_train, X_test, y_train, y_test = load_data()

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()