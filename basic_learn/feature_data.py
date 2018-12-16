import pickle
import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from restaurant_analysis import load_graph

import os, sys
sys.path.append('ml-toolkit')
import util

class DataFeatures:
    def __init__(self, folder='', use_test = False):
        self.raw = load_graph(folder, use_test)
        self.labels = self.raw.review_count.values

        df = self.raw
        feature_cols = {'degree': df.degree, 'clustering': df.clustering, 'comm_edge_density': df.comm_edge_density,
                        'comm_sz': df.comm_sz, 'comm_review_count': df.comm_review_count, 'split': df.split}
        self.feature_matrix = pd.DataFrame(feature_cols)
        self.raw = self.feature_matrix
        self.fname = 'data/ml/graph_features_{}.pkl'.format(folder)

        self.train_indices = self.raw[self.raw.split == 0].index
        self.val_indices = self.raw[self.raw.split == 1].index
        self.test_indices = self.raw[self.raw.split == 2].index
        self.save()

    def get_f_dict(self):
        return dict(zip(util.features, [self.feature_matrix]))

    def get_joint_matrix(self, features):
        f_dict = self.get_f_dict()
        X = [f_dict[f] for f in features]
        X = np.concatenate(tuple(X), axis=1)
        return X

    def save(self):
        util.save_pkl(self.fname, self)


if __name__ == "__main__":
    x = DataFeatures('edge_rem_split', False)

