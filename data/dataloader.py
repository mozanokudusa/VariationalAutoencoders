"""
Data loading
=====================================================================
Supported datasets: MNIST, FashionMNIST, CIFAR10
All images are resized to 32×32 and normalised to [0, 1].
=====================================================================
"""

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


DATASET_CONFIGS: Dict[str, Dict] = {
    "mnist": {
        "cls": torchvision.datasets.MNIST,
        "channels": 1,
        "mean": (0.5,),
        "std": (0.5,),
    },
    "fashion_mnist": {
        "cls": torchvision.datasets.FashionMNIST,
        "channels": 1,
        "mean": (0.5,),
        "std": (0.5,),
    },
    "cifar10": {
        "cls": torchvision.datasets.CIFAR10,
        "channels": 3,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
}

IMAGE_SIZE: int = 32  # uniform resizing


def get_dataset_info(dataset_name: str) -> Dict:
    """Return config dict for a dataset (channels, class, etc.)."""
    key = dataset_name.lower().replace("-", "_")
    if key not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {available}"
        )
    return DATASET_CONFIGS[key]


def _build_transforms(
    channels: int,
    image_size: int = IMAGE_SIZE,
    train: bool = True,
    rotation_degrees: float = 15.0,
    normalize: bool = False,
    mean: Tuple = None,
    std: Tuple = None,
) -> transforms.Compose:
    ops = []

    ops.append(transforms.Resize((image_size, image_size)))

    if train:
        ops.append(transforms.RandomRotation(degrees=rotation_degrees))
        ops.append(transforms.RandomHorizontalFlip(p=0.5))

    ops.append(transforms.ToTensor())  # -> [0, 1]

    if normalize and mean is not None and std is not None:
        ops.append(transforms.Normalize(mean, std))

    return transforms.Compose(ops)


# API
def get_dataloaders(
    dataset_name: str,
    batch_size: int = 128,
    data_root: str = "./data/raw",
    num_workers: int = 4,
    pin_memory: bool = True,
    rotation_degrees: float = 15.0,
    normalize: bool = False,
    image_size: int = IMAGE_SIZE,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train and test DataLoaders for the requested dataset.

    Returns
    -------
    train_loader : DataLoader
    test_loader  : DataLoader
    channels     : int  (1 for grayscale, 3 for RGB)
    """
    cfg = get_dataset_info(dataset_name)
    channels = cfg["channels"]

    train_tf = _build_transforms(
        channels,
        image_size=image_size,
        train=True,
        rotation_degrees=rotation_degrees,
        normalize=normalize,
        mean=cfg["mean"],
        std=cfg["std"],
    )
    test_tf = _build_transforms(
        channels,
        image_size=image_size,
        train=False,
        normalize=normalize,
        mean=cfg["mean"],
        std=cfg["std"],
    )

    DatasetCls = cfg["cls"]
    train_set = DatasetCls(
        root=data_root, train=True, download=True, transform=train_tf
    )
    test_set = DatasetCls(
        root=data_root, train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, test_loader, channels
