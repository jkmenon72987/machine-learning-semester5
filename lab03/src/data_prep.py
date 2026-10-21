
"""
Lab 03: Dataset preparation for architecture and cost comparison.

Loads Fashion-MNIST, normalizes images, and creates reproducible
training, validation, and test datasets.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 128
DATA_DIR = Path("data")
RESULTS_DIR = Path("lab03/results")


def main():
    torch.manual_seed(SEED)

    # Convert image pixels to tensors and normalize them.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    # Download/load Fashion-MNIST.
    train_full = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    # Split 60,000 training images into 55,000 training and 5,000 validation.
    train_dataset, validation_dataset = random_split(
        train_full,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(SEED),
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create loaders for efficient mini-batch processing.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("LAB 03 — DATASET PREPARATION")
    print("-" * 40)
    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Test images:       {len(test_dataset)}")

    images, labels = next(iter(train_loader))

    print(f"\nImage batch shape: {tuple(images.shape)}")
    print(f"Label batch shape: {tuple(labels.shape)}")
    print(f"Normalized pixel range: {images.min():.2f} to {images.max():.2f}")
    print(f"Number of classes: {len(train_full.classes)}")
    print(f"Classes: {train_full.classes}")

    print("\nDataset preparation completed successfully.")
    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    main()

