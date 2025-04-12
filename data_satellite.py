
import torch
import torchvision.transforms as transforms
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import certifi
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def get_satellite_loaders(batch_size, root='./data', resize=64, download=True, subset_classes=None):
    """
    Load EuroSAT dataset with DataLoader

    Args:
        batch_size (int): batch size
        root (str): where to store data
        resize (int): target image size
        download (bool): download if not present
        subset_classes (list[int]): optional class filter (EuroSAT has 10 classes)

    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # EuroSAT RGB mean/std
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = EuroSAT(root=root, transform=transform, download=download)

    if subset_classes is not None:
        indices = [i for i, (_, label) in enumerate(dataset) if label in subset_classes]
        dataset = Subset(dataset, indices)

    # Split 80/20
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
