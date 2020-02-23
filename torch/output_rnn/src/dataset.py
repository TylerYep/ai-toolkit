import sys
import glob
import os
import unicodedata
import string
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from src.metric_tracker import Mode

if 'google.colab' in sys.modules:
    DATA_PATH = '/content/'
else:
    DATA_PATH = 'data/'

INPUT_SHAPE = (1, 19)
ALL_LETTERS = string.ascii_letters + " .,;'"


# def pad_collate(batch):
#     (xx, yy) = zip(*batch)
#     x_lens = [len(x) for x in xx]
#     y_lens = [len(y) for y in yy]
#     xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
#     yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
#     return xx_pad, yy_pad, x_lens, y_lens


def get_collate_fn(device):
    return lambda x: map(lambda b: b.to(device), default_collate(x))


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = torch.tensor([len(x) for x in xx])
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = torch.stack(yy)
    return xx_pad, yy_pad, x_lens


def load_train_data(args, device):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_set = LanguageWords(Mode.TRAIN)
    val_set = LanguageWords(Mode.VAL)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=get_collate_fn(device))
    val_loader = DataLoader(val_set,
                            batch_size=args.test_batch_size,
                            collate_fn=get_collate_fn(device))
    return train_loader, val_loader, class_names, train_set.get_model_params()


def load_test_data(args, device):
    test_set = LanguageWords(Mode.TEST)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             collate_fn=get_collate_fn(device))
    return test_loader


class LanguageWords(Dataset):
    ''' Dataset for training a model on a dataset. '''
    def __init__(self, mode):
        super().__init__()
        self.all_categories = []
        self.data = []
        # Build the category_lines dictionary, a list of names per language
        for filename in glob.glob(f'{DATA_PATH}names/*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.read_lines(filename)
            for word in lines:
                self.data.append((word, category))

        assert self.data
        self.n_categories = len(self.all_categories)
        self.n_letters = len(ALL_LETTERS)
        self.n_hidden = 128
        self.max_word_length = len(max(self.data, key=lambda x: len(x[0]))[0])

        random.shuffle(self.data)
        train_val_split = int(len(self.data) * 0.8)
        if mode == Mode.TRAIN:
            self.data = self.data[:train_val_split]
        elif mode in (Mode.VAL, Mode.TEST):
            self.data = self.data[train_val_split:]

    def get_model_params(self):
        return self.max_word_length, self.n_hidden, self.n_categories  # , self.n_letters

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line, category = self.data[index]
        category_tensor = torch.tensor([self.all_categories.index(category)])
        line_tensor = self.lineToTensor(line)
        # torch.tensor([ALL_LETTERS.index(letter) for letter in line])
        return line_tensor, category_tensor.squeeze()

    def read_lines(self, filename):
        def unicodeToAscii(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS)

        with open(filename, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            return [unicodeToAscii(line) for line in lines]

    def lineToTensor(self, line):
        tensor = torch.zeros((1, self.max_word_length))
        for li, letter in enumerate(line):
            tensor[0][li] = ALL_LETTERS.index(letter)
        return tensor
