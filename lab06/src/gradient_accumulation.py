"""
Lab 06 - Task 6
Compare batch size 256 with batch size 64
using gradient accumulation of 4.
"""

import sys
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import CNN
from data import create_dataloaders, set_seed


def train_model(config, batch_size, accumulation_steps, experiment_name):

    set_seed(config["reproducibility"]["seed"])

    device = torch.device("cpu")

    train_loader, val_loader, _ = create_dataloaders(
        config,
        batch_size=batch_size,
        num_workers=0,
    )

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    history = []

    epochs = 5

    print("\n" + "=" * 60)
    print(experiment_name)
    print("=" * 60)

    start_time = time.perf_counter()

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        accumulated_samples = 0
        accumulated_batches = 0

        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Sum the loss so gradients can be normalized
            # by the total number of samples in each update.
            loss = criterion(outputs, labels)

            loss.backward()

            batch_count = images.size(0)

            accumulated_samples += batch_count
            accumulated_batches += 1

            running_loss += loss.item()
            total_samples += batch_count

            # Update after the requested number of microbatches.
            if accumulated_batches == accumulation_steps:

                for parameter in model.parameters():

                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_samples)

                optimizer.step()
                optimizer.zero_grad()

                accumulated_samples = 0
                accumulated_batches = 0

        # Handle the final incomplete accumulation group.
        if accumulated_batches > 0:

            for parameter in model.parameters():

                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)

            optimizer.step()
            optimizer.zero_grad()

        train_loss = running_loss / total_samples

        # Validation.
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total

        history.append({
            "experiment": experiment_name,
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "validation_accuracy": round(accuracy, 5),
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Validation Accuracy: {accuracy:.4f}"
        )

    elapsed = time.perf_counter() - start_time

    return history, elapsed


def main():

    config_path = PROJECT_ROOT / "lab06/configs/config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Experiment 1: Normal batch size 256.
    normal_history, normal_time = train_model(
        config,
        batch_size=256,
        accumulation_steps=1,
        experiment_name="Batch_256",
    )

    # Experiment 2: Batch size 64, accumulation 4.
    accumulated_history, accumulated_time = train_model(
        config,
        batch_size=64,
        accumulation_steps=4,
        experiment_name="Batch_64_Accumulation_4",
    )

    all_history = normal_history + accumulated_history

    results_dir = PROJECT_ROOT / "lab06/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "gradient_accumulation.csv"

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "epoch",
                "train_loss",
                "validation_accuracy",
            ],
        )

        writer.writeheader()
        writer.writerows(all_history)

    print("\n" + "=" * 60)
    print("GRADIENT ACCUMULATION COMPARISON")
    print("=" * 60)

    print(f"Batch 256 training time: {normal_time:.2f} seconds")
    print(f"Batch 64 + accumulation 4 time: {accumulated_time:.2f} seconds")

    print("\nFinal validation accuracies:")
    print(
        "Batch 256:",
        normal_history[-1]["validation_accuracy"],
    )
    print(
        "Batch 64 + accumulation 4:",
        accumulated_history[-1]["validation_accuracy"],
    )

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()