
"""
Lab 03, Task 5:
Plot model accuracy against parameter count and inference latency.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

TRAINING_FILE = RESULTS_DIR / "training_results.csv"
COST_FILE = RESULTS_DIR / "computational_cost.csv"


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    training_results = read_csv(TRAINING_FILE)
    cost_results = read_csv(COST_FILE)

    training_by_model = {
        row["model"]: row for row in training_results
    }

    cost_by_model = {
        row["model"]: row for row in cost_results
    }

    models = ["MLP", "CNN", "Attention"]

    accuracies = [
        float(training_by_model[name]["test_accuracy"]) * 100
        for name in models
    ]

    parameters = [
        int(cost_by_model[name]["trainable_parameters"])
        for name in models
    ]

    latencies = [
        float(cost_by_model[name]["latency_batch1_median_ms"])
        for name in models
    ]

    # Plot 1: Test accuracy versus parameter count.
    plt.figure(figsize=(8, 5))

    plt.scatter(parameters, accuracies)

    for name, x, y in zip(models, parameters, accuracies):
        plt.annotate(name, (x, y), xytext=(5, 5),
                     textcoords="offset points")

    plt.xlabel("Trainable parameters")
    plt.ylabel("Test accuracy (%)")
    plt.title("Lab 3: Accuracy vs Parameter Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "accuracy_vs_parameters.png",
        dpi=200,
    )
    plt.close()

    # Plot 2: Test accuracy versus single-image inference latency.
    plt.figure(figsize=(8, 5))

    plt.scatter(latencies, accuracies)

    for name, x, y in zip(models, latencies, accuracies):
        plt.annotate(name, (x, y), xytext=(5, 5),
                     textcoords="offset points")

    plt.xlabel("Batch 1 inference latency (ms)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Lab 3: Accuracy vs Inference Latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "accuracy_vs_latency.png",
        dpi=200,
    )
    plt.close()

    print("Lab 3 trade-off plots created successfully.")
    print(f"Saved: {RESULTS_DIR / 'accuracy_vs_parameters.png'}")
    print(f"Saved: {RESULTS_DIR / 'accuracy_vs_latency.png'}")


if __name__ == "__main__":
    main()

