import os
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(split: str, image_size: int) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


def build_samples(raw_dir: str) -> list[tuple[str, int]]:
    """Scan raw_dir for cat/dog images and return (path, label) pairs.

    Expects filenames like 'cat.0.jpg' or 'dog.0.jpg' (Kaggle Dogs vs Cats format).
    Label: cat=0, dog=1.
    """
    samples = []
    for fname in sorted(os.listdir(raw_dir)):
        lower = fname.lower()
        if lower.startswith("cat"):
            label = 0
        elif lower.startswith("dog"):
            label = 1
        else:
            continue
        samples.append((os.path.join(raw_dir, fname), label))
    return samples


def split_samples(
    samples: list[tuple[str, int]],
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[list, list]:
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_split)
    return shuffled[n_val:], shuffled[:n_val]


class CatsDogsDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
