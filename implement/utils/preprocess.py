import numpy as np
import pandas as pd


def train_test_split(array_list, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    """Split data into three subsets (train, validation, and test).

    @param: array_list
            list of np.arrays/torch.Tensors
            we will split each entry accordingly
    @param train_frac: float [default: 0.8]
                       must be within (0.0, 1.0)
    @param val_frac: float [default: 0.8]
                     must be within [0.0, 1.0)
    @param train_frac: float [default: 0.8]
                       must be within (0.0, 1.0)
    """
    assert (train_frac + val_frac + test_frac) == 1.0
    train_list, val_list, test_list = [], [], []

    for arr in array_list:
        size = len(arr)
        start, mid = int(train_frac * size), int((train_frac + val_frac) * size)
        train_list.append(arr[:start])
        if val_frac > 0.0:
            val_list.append(arr[start:mid])
        test_list.append(arr[mid:])

    if val_frac > 0.0:
        return train_list, val_list, test_list

    return train_list, test_list


def split_csv():
    np.random.seed(0)
    df = pd.read_csv("data/train-orig.csv")
    print(f"Dataset size: {len(df)}")

    df_train = df.sample(frac=0.8)
    df_dev = df.drop(df_train.index)
    df_train.to_csv("data/train.csv", index=False)
    df_dev.to_csv("data/dev.csv", index=False)

    print(df_train.nunique())
    print(df_dev.nunique())
    print(pd.concat([df_train, df_dev]).nunique())

    print("Total number of images in train: " + str(len(df_train)))
    print("Total number of images in dev: " + str(len(df_dev)))
