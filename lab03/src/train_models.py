
"""
Lab 03, Task 4:
Train MLP, CNN, and Attention models on Fashion-MNIST.

Configuration:
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 128
- Epochs: 5
- Random seed: 42
"""

import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

# Allow importing modules from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import MLP, CNN, AttentionModel


SEED = 42
LEARNING_RATE = 1e-3
EPOCHS = 5
BATCH_SIZE = 128

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "training_results.csv"


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("LAB 03 — MODEL TRAINING")
    print("=" * 55)
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    # Reuse the existing dataset preparation script.
    from data_prep import main as prepare_data

    train_loader, val_loader, test_loader = prepare_data()

    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}")
        print("-" * 40)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

        start_time = time.perf_counter()

        for epoch in range(EPOCHS):
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

                batch_count = images.size(0)
                running_loss += loss.item() * batch_count
                samples_seen += batch_count

            average_loss = running_loss / samples_seen

            # Measure validation accuracy after each epoch.
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

            validation_accuracy = correct / total

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Loss: {average_loss:.4f} | "
                f"Validation accuracy: {validation_accuracy:.4f}"
            )

        training_time = time.perf_counter() - start_time

        # Evaluate the final model on the official test set.
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        test_accuracy = correct / total

        result = {
            "model": model_name,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "training_time_seconds": round(training_time, 4),
            "test_accuracy": round(test_accuracy, 4),
        }

        results.append(result)

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Test accuracy: {test_accuracy:.4f}")

    # Save all three models' results.
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "epochs",
                "learning_rate",
                "batch_size",
                "training_time_seconds",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print(f"Results saved to: {OUTPUT_FILE}")

    print("\nFinal test accuracy:")
    for result in results:
        print(
            f"{result['model']}: "
            f"{result['test_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()

