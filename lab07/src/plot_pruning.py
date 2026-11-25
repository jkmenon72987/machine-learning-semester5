import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


# Load results
magnitude = pd.read_csv(
    os.path.join(RESULTS_DIR, "magnitude_pruning_results.csv")
)

structured = pd.read_csv(
    os.path.join(RESULTS_DIR, "structured_pruning_results.csv")
)


# ==========================================
# GRAPH 1: GLOBAL VS LAYER-WISE PRUNING
# ==========================================

plt.figure(figsize=(9, 6))

for method in ["global", "layerwise"]:

    data = magnitude[magnitude["method"] == method]

    plt.plot(
        data["actual_sparsity"] * 100,
        data["accuracy_percent"],
        marker="o",
        label=method.capitalize()
    )

plt.xlabel("Weight Sparsity (%)")
plt.ylabel("Test Accuracy (%)")
plt.title("Global vs Layer-wise Magnitude Pruning")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "magnitude_pruning_accuracy.png"),
    dpi=300
)

plt.close()


# ==========================================
# GRAPH 2: STRUCTURED PRUNING
# ==========================================

plt.figure(figsize=(9, 6))

plt.plot(
    structured["actual_filter_sparsity"] * 100,
    structured["accuracy_percent"],
    marker="o",
    label="Structured L2 Filter Pruning"
)

plt.xlabel("Filter Sparsity (%)")
plt.ylabel("Test Accuracy (%)")
plt.title("Structured Pruning Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "structured_pruning_accuracy.png"),
    dpi=300
)

plt.close()


print("Pruning graphs generated successfully!")

print("Saved:")
print("1. magnitude_pruning_accuracy.png")
print("2. structured_pruning_accuracy.png")