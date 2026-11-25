import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from models import OversizedCNN


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

BASELINE_PATH = os.path.join(
    CHECKPOINT_DIR,
    "teacher_baseline.pth"
)

SPARSE_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "global_pruning_95.pth"
)

INPUT = torch.randn(1, 1, 28, 28)


# ==========================================
# 2. LOAD MODELS
# ==========================================

baseline_model = OversizedCNN().to(DEVICE)

baseline_model.load_state_dict(
    torch.load(BASELINE_PATH, map_location=DEVICE)
)

pruned_model = OversizedCNN().to(DEVICE)

pruned_model.load_state_dict(
    torch.load(SPARSE_MODEL_PATH, map_location=DEVICE)
)

baseline_model.eval()
pruned_model.eval()


# ==========================================
# 3. MEASURE INFERENCE LATENCY
# ==========================================

def measure_latency(model, iterations=500):

    model.eval()

    # Warm-up
    with torch.no_grad():

        for _ in range(30):
            model(INPUT)

    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(iterations):
            model(INPUT)

    end = time.perf_counter()

    latency = ((end - start) / iterations) * 1000

    return latency


# ==========================================
# 4. SAVE DENSE CHECKPOINTS
# ==========================================

baseline_dense_path = os.path.join(
    RESULTS_DIR,
    "baseline_dense.pth"
)

pruned_dense_path = os.path.join(
    RESULTS_DIR,
    "pruned_95_dense.pth"
)

torch.save(
    baseline_model.state_dict(),
    baseline_dense_path
)

torch.save(
    pruned_model.state_dict(),
    pruned_dense_path
)


# ==========================================
# 5. CREATE SPARSE STORAGE REPRESENTATION
# ==========================================

sparse_state = {}

for name, tensor in pruned_model.state_dict().items():

    if name.endswith(".weight"):

        sparse_state[name] = tensor.to_sparse_coo()

    else:

        sparse_state[name] = tensor


sparse_path = os.path.join(
    RESULTS_DIR,
    "pruned_95_sparse_representation.pth"
)

torch.save(sparse_state, sparse_path)


# ==========================================
# 6. MEASURE FILE SIZES
# ==========================================

baseline_size = os.path.getsize(baseline_dense_path)

pruned_dense_size = os.path.getsize(pruned_dense_path)

sparse_size = os.path.getsize(sparse_path)


# ==========================================
# 7. MEASURE LATENCY
# ==========================================

print("=" * 60)
print("INFERENCE LATENCY BENCHMARK")
print("=" * 60)

baseline_latency = measure_latency(baseline_model)

pruned_latency = measure_latency(pruned_model)

print(
    f"Baseline CNN latency: "
    f"{baseline_latency:.4f} ms/image"
)

print(
    f"Pruned CNN latency: "
    f"{pruned_latency:.4f} ms/image"
)


# ==========================================
# 8. DISPLAY STORAGE RESULTS
# ==========================================

print("\n" + "=" * 60)
print("MODEL STORAGE COMPARISON")
print("=" * 60)

print(
    f"Baseline dense model: "
    f"{baseline_size / (1024 * 1024):.4f} MB"
)

print(
    f"95% pruned dense model: "
    f"{pruned_dense_size / (1024 * 1024):.4f} MB"
)

print(
    f"95% pruned sparse representation: "
    f"{sparse_size / (1024 * 1024):.4f} MB"
)


# ==========================================
# 9. CALCULATE STORAGE REDUCTION
# ==========================================

storage_reduction = (
    (baseline_size - sparse_size) / baseline_size
) * 100

print(
    f"\nSparse storage reduction: "
    f"{storage_reduction:.2f}%"
)


# ==========================================
# 10. SAVE RESULTS
# ==========================================

results = [
    {
        "model": "Baseline Dense",
        "storage_mb": baseline_size / (1024 * 1024),
        "latency_ms": baseline_latency
    },
    {
        "model": "95% Pruned Dense",
        "storage_mb": pruned_dense_size / (1024 * 1024),
        "latency_ms": pruned_latency
    },
    {
        "model": "95% Pruned Sparse Representation",
        "storage_mb": sparse_size / (1024 * 1024),
        "latency_ms": "Not benchmarked"
    }
]

csv_path = os.path.join(
    RESULTS_DIR,
    "compression_benchmark.csv"
)

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("COMPRESSION BENCHMARK COMPLETE")
print("=" * 60)

print("Results saved:", csv_path)

print("\nNote:")
print(
    "The sparse representation is measured for storage only. "
    "Inference latency is measured using dense PyTorch tensors."
)

print(
    "Unstructured pruning creates zero-valued weights, "
    "but standard dense convolution and linear operations "
    "still process dense tensors."
)