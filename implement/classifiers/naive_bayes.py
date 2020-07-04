import numpy as np

import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
X, y, col_names = data["data"], data["target"], data["feature_names"]
X = pd.DataFrame()

# P(A|B) = P(B|A) * P(A) / P(B)
# https://towardsdatascience.com/implementing-naive-bayes-in-2-minutes-with-python-3ecd788803fe
