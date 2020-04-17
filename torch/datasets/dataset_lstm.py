import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset

if "google.colab" in sys.modules:
    DATA_PATH = "/content/"
else:
    DATA_PATH = "data/names/"


CLASS_LABELS = []


def get_collate_fn(device):
    """
    for indices in batch_sampler:
        yield collate_fn([dataset[i] for i in indices])
    """

    def to_device(b):
        return list(map(to_device, b)) if isinstance(b, (list, tuple)) else b.to(device)

    def sort_batch(data, target):
        batch, lengths = data
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        target_tensor = target[perm_idx]
        return (seq_tensor.transpose(0, 1), seq_lengths), target_tensor

    return lambda x: map(to_device, sort_batch(*default_collate(x)))


def load_train_data(args, device, val_split=0.2):
    collate_fn = get_collate_fn(device)
    orig_dataset = LanguageWords()
    if args.num_examples:
        n = args.num_examples
        data_split = [n, n, len(orig_dataset) - 2 * n]
        train_set, val_set = random_split(orig_dataset, data_split)[:-1]
    else:
        train_size = int((1 - val_split) * len(orig_dataset))
        data_split = [train_size, len(orig_dataset) - train_size]
        train_set, val_set = random_split(orig_dataset, data_split)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_fn)
    return train_loader, val_loader, orig_dataset.get_model_params() + (device,)


def load_test_data(args, device):
    collate_fn = get_collate_fn(device)
    test_set = LanguageWords()
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=collate_fn)
    return test_loader


class LanguageWords(Dataset):
    """Dataset wrapping data, target and length tensors.

    Each sample will be retrieved by indexing both tensors along the first dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        length (Tensor): contains sample lengths.
        raw_data (Any): The data that has been transformed into tensor, useful for debugging
    """

    def __init__(self, data_dir=DATA_PATH):
        super().__init__()
        self.input_shape = [torch.Size((20,)), torch.Size([])]
        self.token2id = defaultdict(int)
        self.token_set, data = self.load_data(data_dir)
        self.token2id = self.set2id(self.token_set, "PAD", "UNK")
        self.tag2id = self.set2id(set(data.keys()))
        self.all_data = []
        for cat in data:
            cat_data = data[cat]
            self.all_data += [(dat, cat) for dat in cat_data]
        # print(data)

        vectorized_seqs = self.vectorized_data(self.all_data, self.token2id)
        self.seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])
        self.data_tensor = self.pad_sequences(vectorized_seqs, self.seq_lengths)
        self.target_tensor = torch.LongTensor([self.tag2id[y] for _, y in self.all_data])

        assert self.data_tensor.size(0) == self.target_tensor.size(0) == self.seq_lengths.size(0)

    def __getitem__(self, index):
        return (self.data_tensor[index], self.seq_lengths[index]), self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

    def get_model_params(self):
        return self.input_shape, len(self.token2id), len(self.tag2id)

    @staticmethod
    def vectorized_data(data, item2id):
        return [
            [item2id[token] if token in item2id else item2id["UNK"] for token in seq]
            for seq, _ in data
        ]

    @staticmethod
    def pad_sequences(vectorized_seqs, seq_lengths):
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        return seq_tensor

    @staticmethod
    def load_data(data_dir):
        filenames = os.listdir(data_dir)
        token_set = set()
        data = defaultdict(list)
        for f in filenames:
            if not f.endswith("txt"):
                continue

            cat = f.replace(".txt", "")
            with open(os.path.join(data_dir, f)) as f:
                for line in f:
                    line = line.strip().lower()
                    data[cat].append(line)
                    for token in line:
                        token_set.add(token)
        return token_set, data

    @staticmethod
    def set2id(item_set, pad=None, unk=None):
        item2id = defaultdict(int)
        if pad is not None:
            item2id[pad] = 0
        if unk is not None:
            item2id[unk] = 1
        for item in item_set:
            item2id[item] = len(item2id)
        return item2id


if __name__ == "__main__":
    x = LanguageWords()
    print(x[0])
