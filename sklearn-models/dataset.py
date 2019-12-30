import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    pima = pd.read_csv("data/diabetes.csv", header=0, names=col_names)

    feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    X = pima[feature_cols]
    y = pima.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return X_train, y_train, X_test, y_test
