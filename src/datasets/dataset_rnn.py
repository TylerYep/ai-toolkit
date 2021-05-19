from __future__ import annotations

import random
import string
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import torch
import wget
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from src.args import Arguments
from src.datasets.dataset import DatasetLoader, TensorDataLoader

DATA_URL = "https://download.pytorch.org/tutorial/data.zip"
ALL_LETTERS = string.ascii_letters + " .,;'"


class DatasetRNN(DatasetLoader):
    @staticmethod
    def pad_collate(batch):
        (xx, yy) = zip(*batch)
        x_lens = torch.tensor([len(x) for x in xx])
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = torch.stack(yy)
        return xx_pad, yy_pad, x_lens

    def load_train_data(
        self, args: Arguments, device: torch.device, val_split: float = 0.2
    ) -> tuple[TensorDataLoader, TensorDataLoader, tuple[Any, ...]]:
        orig_dataset = LanguageWords(self.DATA_PATH)
        train_loader, val_loader = self.split_data(
            orig_dataset, args, device, val_split
        )
        return train_loader, val_loader, orig_dataset.model_params

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


# def pad_collate(batch):
#     (xx, yy) = zip(*batch)
#     x_lens = [len(x) for x in xx]
#     y_lens = [len(y) for y in yy]
#     xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
#     yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
#     return xx_pad, yy_pad, x_lens, y_lens


class LanguageWords(TensorDataset):
    """Dataset for training a model on a dataset."""

    def __init__(self, data_path):
        super().__init__()
        self.input_shape = torch.Size((1, 19))
        self.all_categories = []
        self.data = []

        data_dir = data_path / data_path / "names/"
        if not data_dir.is_dir():
            output_zip = wget.download(DATA_URL, str(data_path))
            with zipfile.ZipFile(output_zip) as zip_ref:
                zip_ref.extractall(data_path)
            Path(output_zip).unlink()

        # Build the category_lines dictionary, a list of names per language
        for filename in data_dir.glob("*.txt"):
            category = filename.stem
            self.all_categories.append(category)
            lines = self.read_lines(filename)
            for word in lines:
                self.data.append((word, category))

        if not self.data:
            raise RuntimeError("Data could not be loaded.")
        self.n_categories = len(self.all_categories)
        self.n_letters = len(ALL_LETTERS)
        self.n_hidden = 128
        self.max_word_length = len(max(self.data, key=lambda x: len(x[0]))[0])

        random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        line, category = self.data[index]
        category_tensor = torch.tensor([self.all_categories.index(category)])
        line_tensor = self.lineToTensor(line)
        # torch.tensor([ALL_LETTERS.index(letter) for letter in line])
        return line_tensor, category_tensor.squeeze()

    @property
    def model_params(self) -> tuple[Any, ...]:
        return self.input_shape, self.max_word_length, self.n_hidden, self.n_categories
        # , self.n_letters

    @staticmethod
    def read_lines(filename: Path) -> list[str]:
        def unicodeToAscii(s: str) -> str:
            return "".join(
                c
                for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
            )

        with open(filename, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
            return [unicodeToAscii(line) for line in lines]

    def lineToTensor(self, line: str) -> torch.Tensor:
        tensor = torch.zeros((1, self.max_word_length))
        for li, letter in enumerate(line):
            tensor[0][li] = ALL_LETTERS.index(letter)
        return tensor
