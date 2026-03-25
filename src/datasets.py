import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(config):
    """
    Standardized Data Loader Factory.
    - Pads MNIST/Fashion-MNIST to 32x32.
    - Applies Horizontal Flips to Fashion-MNIST and CIFAR-10 (Training only).
    """
    dataset_name = config['training'].get('dataset', 'mnist').lower()
    batch_size = config['training'].get('batch_size', 32)
    num_workers = config['training'].get('num_workers', 0)
    data_dir = './data'

    # 1. Define Base Transforms (Applied to both Train and Test)
    # This ensures all datasets arrive at the model as 32x32 [0, 1] tensors
    base_ops = []
    if dataset_name in ['mnist', 'fashion_mnist']:
        base_ops.append(transforms.Pad(2)) # 28x28 -> 32x32
    
    base_ops.append(transforms.ToTensor())
    
    test_transform = transforms.Compose(base_ops)

    # 2. Define Training Transforms (Includes Augmentation)
    train_ops = []
    
    # Apply Horizontal Flip ONLY if semantically valid (Clothing/Objects, not Digits)
    if dataset_name in ['fashion_mnist', 'cifar10']:
        train_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Add the base padding and tensor conversion
    train_ops.extend(base_ops)
    
    train_transform = transforms.Compose(train_ops)

    # 3. Initialize Datasets
    if dataset_name == 'mnist':
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
        test_set = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)
        classes = [str(i) for i in range(10)]

    elif dataset_name == 'fashion_mnist':
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_transform)
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_transform)
        classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    elif dataset_name == 'cifar10':
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 4. Create Loaders
    # We use num_workers=0 for stability as discussed. 
    # pin_memory=True is kept to speed up GPU transfer.
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=max(1000, batch_size), 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    print(f"Loaded {dataset_name.upper()}: Train({len(train_set)}) Test({len(test_set)}) | Final Size: 32x32")
    
    return train_loader, test_loader, classes