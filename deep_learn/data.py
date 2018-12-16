import pandas as pd
import spacy
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import os, sys
sys.path.append('preprocess')
sys.path.append('algorithms')
import read_data
import util


def encode(seq, word2ind):
    return [word2ind[w] if w in word2ind else 0 for w in seq]

def decode(seq, ind2word):
    return [ind2word[w] for w in seq]

def make_vocab_and_dataset():
    wb = read_data.load_weebit()
    doc_objs = util.load_pkl('preprocess/doc_objs.pkl')

    dataset = []
    vocab = Counter()
    for i, doc in enumerate(doc_objs):
        x = [w.pos_ if w.pos_ not in ['PUNCT'] else w.text for w in doc]
        dataset.append(x)
        vocab += Counter(x)

    words = ['UNK'] + [w for w, c in vocab.items() if c > 5]

    word2ind = {w : i for i, w in enumerate(words)}
    ind2word = {i : w for i, w in enumerate(words)}

    dataset = [(encode(x, word2ind), wb.level[i], wb.split[i]) for i, x in enumerate(dataset)]

    util.save_pkl('deep_learn/vocab.pkl', (word2ind, ind2word))
    util.save_pkl('deep_learn/weebit_pos.pkl', dataset)

def make_vocab_real_words():
    wb = read_data.load_weebit()
    dataset = []
    vocab = Counter()
    for i, doc in enumerate(wb.text):
        doc = doc.split()
        doc = [x.lower() for x in doc]
        dataset.append(doc)
        doc = [x for x in doc if x.isalpha()]
        vocab += Counter(doc)

    words = ['UNK'] + [w for w, c in vocab.items() if c > 5]
    word2ind = {w : i for i, w in enumerate(words)}
    ind2word = {i : w for i, w in enumerate(words)}

    dataset = [(encode(x, word2ind), wb.level[i], wb.split[i]) for i, x in enumerate(dataset)]

    util.save_pkl('deep_learn/vocab2.pkl', (word2ind, ind2word))
    util.save_pkl('deep_learn/weebit_words.pkl', dataset)

class Data(Dataset):
    def __init__(self, split, size=None, pos=True):
        # Split 0 = train, 1 = dev, 2 = test

        if pos:
            self.texts = util.load_pkl('deep_learn/weebit_pos.pkl')
        else:
            self.texts = util.load_pkl('deep_learn/weebit_words.pkl')
        self.texts = [x for x in self.texts if x[2] == split]
        if size is not None:
            self.texts = self.texts[:size]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, ix):
        return np.array(self.texts[ix][0]), self.texts[ix][1] - 2

if __name__ == "__main__":

    # make_vocab_and_dataset()
    make_vocab_real_words()
    # vc = util.load_pkl('deep_learn/vocab2.pkl')
    # print(vc)
    # x = Data(0)
    # print(x[0])

