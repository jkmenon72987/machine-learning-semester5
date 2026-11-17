"""
Lab 06 - Task 2
Profile one CNN training epoch using torch.profiler.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import CNN
from data import create_dataloaders, set_seed


def main():

    config_path = PROJECT_ROOT / "lab06/configs/config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    set_seed(config["reproducibility"]["seed"])

    device = torch.device("cpu")

    train_loader, _, _ = create_dataloaders(config)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    model.train()

    activities = [ProfilerActivity.CPU]

    timing = {
        "data_loading": 0.0,
        "forward_pass": 0.0,
        "backward_pass": 0.0,
        "optimizer_step": 0.0,
    }

    print("Starting one-epoch profiling...")
    start_time = time.perf_counter()

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:

        iterator = iter(train_loader)

        while True:

            # Measure data loading.
            try:
                data_start = time.perf_counter()

                with record_function("data_loading"):
                    images, labels = next(iterator)

                timing["data_loading"] += (
                    time.perf_counter() - data_start
                )

            except StopIteration:
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Measure forward pass.
            forward_start = time.perf_counter()

            with record_function("forward_pass"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            timing["forward_pass"] += (
                time.perf_counter() - forward_start
            )

            # Measure backward pass.
            backward_start = time.perf_counter()

            with record_function("backward_pass"):
                loss.backward()

            timing["backward_pass"] += (
                time.perf_counter() - backward_start
            )

            # Measure optimizer step.
            optimizer_start = time.perf_counter()

            with record_function("optimizer_step"):
                optimizer.step()

            timing["optimizer_step"] += (
                time.perf_counter() - optimizer_start
            )

    total_time = time.perf_counter() - start_time

    print("\nPROFILING COMPLETED")
    print("=" * 55)

    measured_total = sum(timing.values())

    for name, seconds in timing.items():

        percentage = (
            seconds / measured_total * 100
            if measured_total > 0 else 0
        )

        print(
            f"{name:20s}: "
            f"{seconds:.4f} seconds "
            f"({percentage:.2f}%)"
        )

    print("-" * 55)
    print(f"Total measured time: {measured_total:.4f} seconds")
    print(f"Overall profiler duration: {total_time:.4f} seconds")

    # Save profiler trace.
    results_dir = PROJECT_ROOT / "lab06/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    trace_path = results_dir / "profiler_trace.json"

    prof.export_chrome_trace(str(trace_path))

    # Save profiler operator summary.
    summary_path = results_dir / "profiler_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=20,
        ))

    print("\nProfiler trace saved:", trace_path)
    print("Profiler summary saved:", summary_path)


if __name__ == "__main__":
    main()