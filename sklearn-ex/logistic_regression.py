from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from dataset import load_data


def main():
    X_train, X_test, y_train, y_test = load_data()

    clf = LogisticRegression()
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()
