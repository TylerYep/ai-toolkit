import pandas as pd
import scipy.stats
import numpy as np
import util
import itertools
from tqdm import tqdm

import os, sys
sys.path.append('ml-toolkit')
import algs
from algs import Algorithm
from read_data import DataFeatures

def get_results(alg: Algorithm, data, feature_lists, options_c, options_wc, options_tfidf):
    prod = itertools.product(feature_lists, options_c, options_wc, options_tfidf)
    for f, c, wc, t in tqdm(list(prod)):
        if not('tfidf' in f or t == {}): continue # Ignore the case where theres not tfidf in features but tfdif is not {}
        if not('word count' in f or wc == {}): continue
        alg.run(data, f, c, wc, t)

def bit_twiddle_params(a, data, features):
    a.run(data, features)#, clf_options=options, wc_params={'min_df':5, 'max_df':0.8, 'binary':True})
    a.to_csv()

if __name__ == "__main__":
    feature_path = 'data/ml/graph_features.pkl'
    x = DataFeatures(folder, True)
    for name in names:
        opts = {}
        if name == "Decision_Tree":
            opts = {'min_samples_split': 20, 'max_features': 'log2', 'min_samples_leaf': 20}
        name += "-extra"
        a = algs.load_alg(name)
        data = util.load_pkl(feature_path)

        a.run(data, util.features, clf_options=opts)
        a.to_csv()
