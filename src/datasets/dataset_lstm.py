from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import TensorDataset

from src.args import Arguments
from src.datasets.dataset import DatasetLoader, TensorDataLoader


class DatasetLSTM(DatasetLoader):
    @staticmethod
    def get_collate_fn(device: torch.device) -> Callable[[list[Any]], Any]:
        """
        for indices in batch_sampler:
            yield collate_fn([dataset[i] for i in indices])
        """

        def to_device(b):
            return (
                list(map(to_device, b))
                if isinstance(b, (list, tuple))
                else b.to(device)
            )

        def sort_batch(data, target):
            batch, lengths = data
            seq_lengths, perm_idx = lengths.sort(0, descending=True)
            seq_tensor = batch[perm_idx]
            target_tensor = target[perm_idx]
            return (seq_tensor.transpose(0, 1), seq_lengths), target_tensor

        return lambda x: map(to_device, sort_batch(*default_collate(x)))

    def load_train_data(
        self, args: Arguments, device: torch.device, val_split: float = 0.2
    ) -> tuple[TensorDataLoader, TensorDataLoader, tuple[Any, ...]]:
        orig_dataset = LanguageWords(self.DATA_PATH)
        train_loader, val_loader = self.split_data(
            orig_dataset, args, device, val_split
        )
        return train_loader, val_loader, orig_dataset.model_params + (device,)

    def load_test_data(self, args: Arguments, device: torch.device) -> TensorDataLoader:
        collate_fn = self.get_collate_fn(device)
        test_set = LanguageWords(self.DATA_PATH)
        test_loader = DataLoader(
            test_set,
            batch_size=args.test_batch_size,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            num_workers=args.num_workers,
        )
        return test_loader


class LanguageWords(TensorDataset):
    """Dataset wrapping data, target and length tensors.

    Each sample will be retrieved by indexing both tensors along the first dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        length (Tensor): contains sample lengths.
        raw_data (Any): The data that has been transformed into
            tensor, useful for debugging
    """

    def __init__(self, data_dir):
        super().__init__()
        self.input_shape = [torch.Size((20,)), torch.Size([])]
        self.token2id = defaultdict(int)
        self.token_set, data = self.load_data(data_dir)
        self.token2id = self.set2id(self.token_set, "PAD", "UNK")
        self.tag2id = self.set2id(set(data))
        self.all_data = []
        for cat in data:
            cat_data = data[cat]
            self.all_data += [(dat, cat) for dat in cat_data]

        vectorized_seqs = self.vectorized_data(self.all_data, self.token2id)
        self.seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])
        self.data_tensor = self.pad_sequences(vectorized_seqs, self.seq_lengths)
        self.target_tensor = torch.LongTensor(
            [self.tag2id[y] for _, y in self.all_data]
        )

        if not (
            self.data_tensor.size(0)
            == self.target_tensor.size(0)
            == self.seq_lengths.size(0)
        ):
            raise RuntimeError("Sizes do not match.")

    def __getitem__(self, index):
        return (self.data_tensor[index], self.seq_lengths[index]), self.target_tensor[
            index
        ]

    def __len__(self):
        return self.data_tensor.size(0)

    @property
    def model_params(self):
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
        token_set = set()
        data = defaultdict(list)
        for filepath in Path(data_dir).glob("*.txt"):
            with open(filepath) as f:
                for line in f:
                    line = line.strip().lower()
                    data[filepath.stem].append(line)
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
