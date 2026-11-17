"""
Lab 06 - Task 3
Compare DataLoader worker counts.
"""

import sys
import time
import csv
from pathlib import Path

import torch
import yaml

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():

    config_path = PROJECT_ROOT / "lab06/configs/config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["reproducibility"]["seed"]
    batch_size = config["training"]["batch_size"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (config["dataset"]["normalization"]["mean"],),
            (config["dataset"]["normalization"]["std"],)
        )
    ])

    dataset = datasets.FashionMNIST(
        root=config["dataset"]["root"],
        train=True,
        download=True,
        transform=transform
    )

    # Use the same fixed 10,000 samples for every test.
    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    )[:10000].tolist()

    subset = Subset(dataset, indices)

    worker_counts = [0, 1, 2, 4, 8]

    results = []

    print("LAB 06 - DATALOADER WORKER BENCHMARK")
    print("=" * 55)
    print("Samples per test: 10,000")
    print("Batch size:", batch_size)

    for workers in worker_counts:

        print(f"\nTesting num_workers = {workers}")

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=False
        )

        # Warm-up: load a few batches before timing.
        iterator = iter(loader)

        for _ in range(min(5, len(loader))):
            try:
                next(iterator)
            except StopIteration:
                break

        # Measure a complete pass over the same dataset.
        start = time.perf_counter()

        samples = 0

        for images, labels in loader:
            samples += images.size(0)

        elapsed = time.perf_counter() - start

        throughput = samples / elapsed

        result = {
            "num_workers": workers,
            "samples": samples,
            "time_seconds": round(elapsed, 4),
            "throughput_samples_per_second": round(
                throughput, 2
            )
        }

        results.append(result)

        print(f"Time: {elapsed:.2f} seconds")
        print(f"Throughput: {throughput:.2f} images/second")

        del loader

    # Save results.
    results_dir = PROJECT_ROOT / "lab06/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "worker_benchmark.csv"

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "num_workers",
                "samples",
                "time_seconds",
                "throughput_samples_per_second"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 55)
    print("WORKER BENCHMARK COMPLETED")

    for result in results:
        print(result)

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()