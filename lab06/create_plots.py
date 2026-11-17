import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# --------------------------------------------------
# 1. DataLoader worker comparison
# --------------------------------------------------

worker_file = RESULTS_DIR / "worker_benchmark.csv"

if worker_file.exists():
    df = pd.read_csv(worker_file)
    print("\nWorker benchmark columns:", list(df.columns))

    x = next(c for c in df.columns if "worker" in c.lower())
    y = next(c for c in df.columns if "throughput" in c.lower())

    plt.figure(figsize=(8, 5))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel("Number of DataLoader Workers")
    plt.ylabel("Throughput (images/second)")
    plt.title("DataLoader Worker Performance")
    plt.grid(True)

    save_plot("worker_throughput.png")


# --------------------------------------------------
# 2. Batch size throughput
# --------------------------------------------------

batch_file = RESULTS_DIR / "batch_size_benchmark.csv"

if batch_file.exists():
    df = pd.read_csv(batch_file)
    print("\nBatch benchmark columns:", list(df.columns))

    x = next(c for c in df.columns if "batch" in c.lower())
    y = next(c for c in df.columns if "throughput" in c.lower())

    plt.figure(figsize=(8, 5))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (images/second)")
    plt.title("Batch Size vs Training Throughput")
    plt.grid(True)

    save_plot("batch_throughput.png")


# --------------------------------------------------
# 3. Batch size vs peak memory
# --------------------------------------------------

if batch_file.exists():
    df = pd.read_csv(batch_file)

    x = next(c for c in df.columns if "batch" in c.lower())
    y = next(c for c in df.columns if "memory" in c.lower())

    plt.figure(figsize=(8, 5))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Peak Memory (MB)")
    plt.title("Batch Size vs Peak Memory")
    plt.grid(True)

    save_plot("batch_memory.png")


# --------------------------------------------------
# 4. Batch size vs validation accuracy
# --------------------------------------------------

if batch_file.exists():
    df = pd.read_csv(batch_file)

    x = next(c for c in df.columns if "batch" in c.lower())
    y = next(
        c for c in df.columns
        if "accuracy" in c.lower()
    )

    plt.figure(figsize=(8, 5))
    plt.plot(df[x], df[y] * 100, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Batch Size vs Validation Accuracy")
    plt.grid(True)

    save_plot("batch_accuracy.png")


# --------------------------------------------------
# 5. Optimizer and scheduler comparison
# --------------------------------------------------

optimizer_file = RESULTS_DIR / "optimizer_scheduler_history.csv"

if optimizer_file.exists():
    df = pd.read_csv(optimizer_file)
    print("\nOptimizer history columns:", list(df.columns))

    epoch = next(c for c in df.columns if "epoch" in c.lower())
    accuracy = next(
        c for c in df.columns
        if "accuracy" in c.lower()
    )

    group_columns = [
        c for c in df.columns
        if c.lower() in ["optimizer", "scheduler"]
    ]

    plt.figure(figsize=(10, 6))

    for name, group in df.groupby(group_columns):
        label = " + ".join(map(str, name if isinstance(name, tuple) else [name]))
        plt.plot(
            group[epoch],
            group[accuracy] * 100,
            marker="o",
            label=label
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Optimizer and Scheduler Comparison")
    plt.legend()
    plt.grid(True)

    save_plot("optimizer_scheduler_comparison.png")


# --------------------------------------------------
# 6. Gradient accumulation comparison
# --------------------------------------------------

gradient_file = RESULTS_DIR / "gradient_accumulation.csv"

if gradient_file.exists():
    df = pd.read_csv(gradient_file)
    print("\nGradient accumulation columns:", list(df.columns))

    print("Gradient accumulation data:")
    print(df.head())

    epoch = next(c for c in df.columns if "epoch" in c.lower())

    accuracy_columns = [
        c for c in df.columns
        if "accuracy" in c.lower()
    ]

    plt.figure(figsize=(9, 5))

    for column in accuracy_columns:
        plt.plot(
            df[epoch],
            df[column] * 100,
            marker="o",
            label=column
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Gradient Accumulation vs Normal Batch Training")
    plt.legend()
    plt.grid(True)

    save_plot("gradient_accumulation_comparison.png")


# --------------------------------------------------
# 7. Profiler breakdown
# --------------------------------------------------

profiler_labels = [
    "Data Loading",
    "Forward Pass",
    "Backward Pass",
    "Optimizer Step"
]

profiler_times = [
    30.3499,
    8.2078,
    10.5481,
    0.9086
]

plt.figure(figsize=(9, 5))
plt.bar(profiler_labels, profiler_times)
plt.ylabel("Time (seconds)")
plt.title("Training Pipeline Time Breakdown")
plt.xticks(rotation=15)
plt.grid(axis="y")

save_plot("profiler_breakdown.png")


print("\nAll available plots have been generated!")
print("Plot location:", PLOTS_DIR)