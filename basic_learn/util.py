import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

features = ['graph_features']

results_headers = ['model_type', 'clf_options', 'train_acc', 'dev_acc', 'test_acc']

names = ["k_Nearest_Neighbors", "SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "Logistic_Regression", 'Dummy', 'LinearReg']

models = [KNeighborsClassifier, SVC, GaussianProcessClassifier, DecisionTreeClassifier,
        RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB,
        LogisticRegression, DummyClassifier, LinearRegression]

model_dict = dict(zip(names, models))

def get_acc(true, pred):
    return(np.mean(true == pred))

def save_pkl(fname, obj):
    with open(fname, 'wb+') as f:
        pickle.dump(obj, f)

def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
