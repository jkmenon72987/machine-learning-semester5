"""
Lab 06: Fashion-MNIST dataset preparation.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(config, batch_size=None, num_workers=None):

    dataset_config = config["dataset"]
    training_config = config["training"]
    seed = config["reproducibility"]["seed"]

    if batch_size is None:
        batch_size = training_config["batch_size"]

    if num_workers is None:
        num_workers = training_config["num_workers"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (dataset_config["normalization"]["mean"],),
            (dataset_config["normalization"]["std"],)
        )
    ])

    train_full = datasets.FashionMNIST(
        root=dataset_config["root"],
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=dataset_config["root"],
        train=False,
        download=True,
        transform=transform
    )

    train_dataset, validation_dataset = random_split(
        train_full,
        [
            dataset_config["train_size"],
            dataset_config["validation_size"]
        ],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, validation_loader, test_loader