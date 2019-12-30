import pandas as pd
import numpy as np

np.random.seed(0)
df = pd.read_csv('data/train-orig.csv')
print(f'Dataset size: {len(df)}')

df_train = df.sample(frac=0.8)
df_dev = df.drop(df_train.index)
df_train.to_csv('data/train.csv', index=False)
df_dev.to_csv('data/dev.csv', index=False)

print(df_train.nunique())
print(df_dev.nunique())
print(pd.concat([df_train, df_dev]).nunique())

print("Total number of images in train: " + str(len(df_train)))
print("Total number of images in dev: " + str(len(df_dev)))
