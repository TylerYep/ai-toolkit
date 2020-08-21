# type: ignore
# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import os
import random
from typing import Any, Callable, List, Tuple

import numpy as np
import torch
from PIL import Image
from src.args import Arguments
from src.datasets.dataset import DatasetLoader
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as F


class DatasetPenn(DatasetLoader):
    def __init__(self):
        super().__init__()
        self.CLASS_LABELS = ["YES", "NO"]

    def load_train_data(
        self, args: Arguments, device: torch.device, val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, Tuple[Any, ...]]:
        orig_dataset = PennFudanDataset("data", self.get_transforms(train=True))
        train_loader, val_loader = self.split_data(orig_dataset, args, device, val_split)
        init_params = []
        return train_loader, val_loader, init_params

    def load_test_data(self, args: Arguments, device: torch.device) -> DataLoader:
        collate_fn = self.get_collate_fn(device)
        test_set = PennFudanDataset("data", self.get_transforms(train=False))
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=collate_fn)
        return test_loader

    @staticmethod
    def get_collate_fn(device: torch.device) -> Callable[[List[Any]], Any]:
        def collate(x):
            data, target = tuple(zip(*x))
            data = list(image.to(device) for image in data)
            target = [{k: v.to(device) for k, v in t.items()} for t in target]
            return (data, target)

        return collate

    @staticmethod
    def get_transforms(train):
        if train:
            return Compose([ToTensor(), RandomHorizontalFlip(0.5)])
        return Compose([ToTensor()])


# output = model(data, target)
# loss = sum(l for l in output.values())


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class PennFudanDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
