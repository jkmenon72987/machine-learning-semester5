
"""
Lab 03, Task 3:
Parameter count, hand-calculated FLOPs, THOP profiling,
and inference latency measurements.
"""

import csv
import statistics
import sys
import time
from pathlib import Path

import torch
from thop import profile
from torchinfo import summary

# Allow importing models.py from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import MLP, CNN, AttentionModel


SEED = 42
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def count_parameters(model):
    """Count trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def measure_latency(model, batch_size, runs):
    """Measure median inference latency in milliseconds."""
    model.eval()
    sample = torch.randn(batch_size, 1, 28, 28)

    # Warm-up runs.
    with torch.no_grad():
        for _ in range(10):
            model(sample)

        timings = []

        for _ in range(runs):
            start = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - start) * 1000)

    return statistics.median(timings)


def hand_calculated_flops():
    """
    Estimate forward-pass FLOPs for one image.

    Convention:
    - One multiply and one addition count as two FLOPs.
    - Bias additions, activations, pooling, softmax, and normalization
      are excluded from these matrix/convolution arithmetic estimates.
    - Transformer attention matrix products are included explicitly.
    """

    # MLP: Linear layers 784->256, 256->128, 128->10.
    mlp = 2 * (
        784 * 256
        + 256 * 128
        + 128 * 10
    )

    # CNN:
    # Conv1: 1->16 channels, 3x3 kernel, 28x28 output.
    conv1 = 2 * 3 * 3 * 1 * 16 * 28 * 28

    # Conv2: 16->32 channels, 3x3 kernel, 14x14 output.
    conv2 = 2 * 3 * 3 * 16 * 32 * 14 * 14

    # Classifier: 32*7*7 -> 10.
    cnn_linear = 2 * (32 * 7 * 7) * 10

    cnn = conv1 + conv2 + cnn_linear

    # Attention model:
    # 16 patches, 49 input values per patch, embedding size 64.
    num_patches = 16
    embedding = 64
    feedforward = 128
    heads = 4

    # Patch projection: 16 x (49 -> 64).
    patch_projection = 2 * num_patches * 49 * embedding

    # Each Transformer encoder layer:
    # Q, K, V projections and output projection.
    qkv_and_output = 8 * num_patches * embedding * embedding

    # Attention scores QK^T and attention-weighted values AV.
    attention_products = 4 * num_patches * num_patches * embedding

    # Feed-forward network: 64 -> 128 -> 64.
    feedforward_flops = (
        2 * num_patches * embedding * feedforward
        + 2 * num_patches * feedforward * embedding
    )

    transformer_layer = (
        qkv_and_output
        + attention_products
        + feedforward_flops
    )

    # Two Transformer encoder layers and final 64->10 classifier.
    attention = (
        patch_projection
        + 2 * transformer_layer
        + 2 * embedding * 10
    )

    return {
        "MLP": mlp,
        "CNN": cnn,
        "Attention": attention,
    }


def main():
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    hand_flops = hand_calculated_flops()
    rows = []

    print("LAB 03 — COMPUTATIONAL COST")
    print("=" * 65)
    print("FLOPs are estimates for one image using the documented convention.")

    for name, model in models.items():
        model.eval()
        parameters = count_parameters(model)
        sample = torch.randn(1, 1, 28, 28)

        # THOP profiling.
        try:
            macs, _ = profile(
                model,
                inputs=(sample,),
                verbose=False,
            )
            thop_flops = 2 * macs
        except Exception as exc:
            print(f"\nTHOP profiling failed for {name}: {exc}")
            macs = float("nan")
            thop_flops = float("nan")

        manual_flops = hand_flops[name]

        # Model summary for the requested torchinfo inspection.
        summary(
            model,
            input_size=(1, 1, 28, 28),
            verbose=0,
        )

        # Batch 1: 100 timed runs.
        latency_batch_1 = measure_latency(model, 1, 100)

        # Batch 128: 20 timed runs.
        latency_batch_128 = measure_latency(model, 128, 20)

        print(f"\n{name}")
        print(f"Trainable parameters: {parameters:,}")
        print(f"Hand-calculated FLOPs: {manual_flops:,}")
        print(f"THOP MACs: {macs:,.0f}")
        print(f"THOP estimated FLOPs (2 x MACs): {thop_flops:,.0f}")
        print(f"Batch 1 median latency: {latency_batch_1:.4f} ms")
        print(f"Batch 128 median latency: {latency_batch_128:.4f} ms")

        rows.append(
            {
                "model": name,
                "trainable_parameters": parameters,
                "hand_calculated_flops_batch1": manual_flops,
                "thop_macs_batch1": macs,
                "thop_flops_batch1": thop_flops,
                "latency_batch1_median_ms": latency_batch_1,
                "latency_batch128_median_ms": latency_batch_128,
            }
        )

    output_file = RESULTS_DIR / "computational_cost.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 65)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

