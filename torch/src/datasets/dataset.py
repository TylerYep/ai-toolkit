import sys

from torch.utils.data import DataLoader, random_split


class DatasetLoader:
    def __init__(self):
        self.CLASS_LABELS = []
        self.DATA_PATH = "/content/" if "google.colab" in sys.modules else "data/"

    def split_data(self, orig_dataset, args, device, val_split):
        collate_fn = self.get_collate_fn(device)
        if args.num_examples:
            n = args.num_examples
            data_split = [n, n, len(orig_dataset) - 2 * n]
            train_set, val_set = random_split(orig_dataset, data_split)[:-1]
        else:
            data_split = [int(part * len(orig_dataset)) for part in (1 - val_split, val_split)]
            train_set, val_set = random_split(orig_dataset, data_split)

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_fn)
        return train_loader, val_loader

    def load_train_data(self, args, device, val_split=0.2):
        raise NotImplementedError

    def load_test_data(self, args, device):
        raise NotImplementedError

    @staticmethod
    def get_collate_fn(device):
        raise NotImplementedError

    @staticmethod
    def get_transforms(img_dim=None):
        raise NotImplementedError
