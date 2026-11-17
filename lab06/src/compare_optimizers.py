"""
Lab 06 - Task 5
Compare SGD and Adam with three learning-rate schedulers.
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

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)


def evaluate(model, loader, criterion, device):

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def run_experiment(config, optimizer_name, scheduler_name):

    set_seed(config["reproducibility"]["seed"])

    device = torch.device("cpu")

    train_loader, val_loader, _ = create_dataloaders(
        config,
        batch_size=128,
        num_workers=0,
    )

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001

    if optimizer_name == "Adam":

        optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
        )

    else:

        optimizer = SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
        )

    epochs = 10

    if scheduler_name == "StepLR":

        scheduler = StepLR(
            optimizer,
            step_size=3,
            gamma=0.1,
        )

    elif scheduler_name == "CosineAnnealingLR":

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )

    else:

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=2,
        )

    history = []

    print("\n" + "=" * 60)
    print(f"Optimizer: {optimizer_name}")
    print(f"Scheduler: {scheduler_name}")
    print("=" * 60)

    start_time = time.perf_counter()

    for epoch in range(epochs):

        model.train()

        running_loss = 0
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

        train_loss = running_loss / samples_seen

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "validation_loss": round(val_loss, 5),
            "validation_accuracy": round(val_accuracy, 5),
            "learning_rate": current_lr,
        }

        history.append(record)

        print(
            f"Epoch {epoch + 1}/10 | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    elapsed = time.perf_counter() - start_time

    return history, elapsed


def main():

    config_path = PROJECT_ROOT / "lab06/configs/config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    optimizers = ["Adam", "SGD"]
    schedulers = [
        "StepLR",
        "CosineAnnealingLR",
        "ReduceLROnPlateau",
    ]

    results_dir = PROJECT_ROOT / "lab06/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    history_file = results_dir / "optimizer_scheduler_history.csv"
    summary_file = results_dir / "optimizer_scheduler_summary.csv"

    all_history = []
    summaries = []

    for optimizer_name in optimizers:

        for scheduler_name in schedulers:

            history, elapsed = run_experiment(
                config,
                optimizer_name,
                scheduler_name,
            )

            all_history.extend(history)

            best_record = max(
                history,
                key=lambda row: row["validation_accuracy"],
            )

            summary = {
                "optimizer": optimizer_name,
                "scheduler": scheduler_name,
                "training_time_seconds": round(elapsed, 2),
                "final_validation_accuracy":
                    history[-1]["validation_accuracy"],
                "best_validation_accuracy":
                    best_record["validation_accuracy"],
                "best_epoch": best_record["epoch"],
                "final_validation_loss":
                    history[-1]["validation_loss"],
            }

            summaries.append(summary)

            # Save progress after each experiment.
            with history_file.open(
                "w", newline="", encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=list(all_history[0].keys()),
                )
                writer.writeheader()
                writer.writerows(all_history)

            with summary_file.open(
                "w", newline="", encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=list(summaries[0].keys()),
                )
                writer.writeheader()
                writer.writerows(summaries)

    print("\n" + "=" * 60)
    print("OPTIMIZER AND SCHEDULER COMPARISON COMPLETED")
    print("=" * 60)

    for result in summaries:
        print(result)

    print("\nHistory:", history_file)
    print("Summary:", summary_file)


if __name__ == "__main__":
    main()