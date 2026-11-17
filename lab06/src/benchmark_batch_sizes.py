"""
Lab 06 - Task 4
Batch-size performance comparison.

Measures:
- Training throughput
- Peak CPU memory usage
- Final validation accuracy
"""

import sys
import csv
import time
import threading
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import CNN
from data import create_dataloaders, set_seed


class MemoryMonitor:
    """Monitor peak process memory usage."""

    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.running = False

    def monitor(self):
        while self.running:
            memory = self.process.memory_info().rss / (1024 ** 2)
            self.peak_memory = max(self.peak_memory, memory)
            time.sleep(0.05)

    def start(self):
        self.running = True
        self.peak_memory = (
            self.process.memory_info().rss / (1024 ** 2)
        )
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        return self.peak_memory


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_one_configuration(config, batch_size):

    seed = config["reproducibility"]["seed"]
    set_seed(seed)

    device = torch.device("cpu")

    train_loader, val_loader, _ = create_dataloaders(
        config,
        batch_size=batch_size,
        num_workers=0,
    )

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    epochs = config["training"]["epochs"]

    monitor = MemoryMonitor()
    monitor.start()

    start_time = time.perf_counter()

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        samples_seen = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            samples_seen += images.size(0)

        average_loss = running_loss / samples_seen

        print(
            f"Batch {batch_size} | "
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f}"
        )

    elapsed = time.perf_counter() - start_time

    peak_memory = monitor.stop()

    # Calculate validation accuracy after training.
    accuracy = evaluate(model, val_loader, device)

    throughput = (
        len(train_loader.dataset) * epochs / elapsed
    )

    return {
        "batch_size": batch_size,
        "epochs": epochs,
        "training_time_seconds": round(elapsed, 2),
        "throughput_samples_per_second": round(throughput, 2),
        "peak_memory_mb": round(peak_memory, 2),
        "validation_accuracy": round(accuracy, 4),
    }


def main():

    config_path = PROJECT_ROOT / "lab06/configs/config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    batch_sizes = [32, 64, 128, 256, 512]

    results = []

    print("=" * 65)
    print("LAB 06 - BATCH SIZE BENCHMARK")
    print("=" * 65)
    print("Device: CPU")
    print("Epochs:", config["training"]["epochs"])
    print("Batch sizes:", batch_sizes)

    for batch_size in batch_sizes:

        print("\n" + "-" * 55)
        print(f"Starting batch size: {batch_size}")

        result = train_one_configuration(
            config,
            batch_size,
        )

        results.append(result)

        print("\nConfiguration results:")
        for key, value in result.items():
            print(f"{key}: {value}")

        # Save results after every completed configuration.
        output_file = (
            PROJECT_ROOT / "lab06/results/batch_size_benchmark.csv"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(results)

    print("\n" + "=" * 65)
    print("BATCH SIZE BENCHMARK COMPLETED")
    print("=" * 65)

    for result in results:
        print(result)

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()