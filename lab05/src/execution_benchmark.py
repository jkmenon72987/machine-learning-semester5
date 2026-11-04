import os
import time
import statistics

import torch
import torch.nn as nn


# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = "cpu"

WARMUP_RUNS = 5
BENCHMARK_RUNS = 20

torch.manual_seed(42)

print("\nLAB 5 - TASK 6")
print("==============================")

print("PyTorch version:", torch.__version__)
print("Device:", DEVICE)


# ==========================================
# 1. SMALL NEURAL NETWORK
# ==========================================

class SmallNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):

        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))

        return x


model = SmallNetwork().to(DEVICE)

sample_input = torch.randn(1, 2)


# ==========================================
# 2. INSPECT COMPUTATIONAL GRAPH
# ==========================================

print("\n1. COMPUTATIONAL GRAPH")
print("------------------------------")

traced_model = torch.jit.trace(model, sample_input)

print(traced_model.graph)


# ==========================================
# 3. INSPECT TENSOR PROPERTIES
# ==========================================

print("\n2. TENSOR PROPERTIES")
print("------------------------------")

tensor = torch.randn(512, 512)

print("Original tensor:")
print("Shape:", tensor.shape)
print("Data type:", tensor.dtype)
print("Contiguous:", tensor.is_contiguous())
print("Stride:", tensor.stride())


transposed_tensor = tensor.t()

print("\nTransposed tensor:")
print("Shape:", transposed_tensor.shape)
print("Data type:", transposed_tensor.dtype)
print("Contiguous:", transposed_tensor.is_contiguous())
print("Stride:", transposed_tensor.stride())


contiguous_tensor = transposed_tensor.contiguous()

print("\nConverted contiguous tensor:")
print("Shape:", contiguous_tensor.shape)
print("Data type:", contiguous_tensor.dtype)
print("Contiguous:", contiguous_tensor.is_contiguous())
print("Stride:", contiguous_tensor.stride())


# ==========================================
# 4. BENCHMARK FUNCTION
# ==========================================

def benchmark(function, input_tensor):

    # Warm-up
    for _ in range(WARMUP_RUNS):
        function(input_tensor)

    times = []

    # Repeated measurements
    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        function(input_tensor)

        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return statistics.median(times)


# ==========================================
# 5. EAGER EXECUTION
# ==========================================

print("\n3. EAGER EXECUTION")
print("------------------------------")

model.eval()

with torch.no_grad():

    eager_time = benchmark(model, sample_input)

print("Eager median time:", eager_time)


# ==========================================
# 6. COMPILED OR TRACED EXECUTION
# ==========================================

print("\n4. COMPILED / TRACED EXECUTION")
print("------------------------------")

compiled_model = None
execution_mode = None

try:

    print("Attempting torch.compile()...")

    compiled_model = torch.compile(model)

    with torch.no_grad():

        # Warm up compilation
        for _ in range(WARMUP_RUNS):
            compiled_model(sample_input)

        compiled_time = benchmark(
            compiled_model,
            sample_input
        )

    execution_mode = "torch.compile"

except Exception as error:

    print("torch.compile() was unavailable or failed.")
    print("Reason:", str(error)[:300])

    print("\nUsing torch.jit.trace() instead.")

    compiled_model = torch.jit.trace(model, sample_input)

    with torch.no_grad():

        traced_time = benchmark(
            compiled_model,
            sample_input
        )

    compiled_time = traced_time

    execution_mode = "torch.jit.trace"


print("Execution mode:", execution_mode)
print("Eager median:", eager_time)
print("Compiled/traced median:", compiled_time)


# ==========================================
# 7. CONTIGUOUS VS NON-CONTIGUOUS
# ==========================================

print("\n5. TENSOR PERFORMANCE")
print("------------------------------")


def tensor_operation(x):

    return x * 1.5 + 2.0


with torch.no_grad():

    non_contiguous_time = benchmark(
        tensor_operation,
        transposed_tensor
    )

    contiguous_time = benchmark(
        tensor_operation,
        contiguous_tensor
    )


print("Non-contiguous tensor median:", non_contiguous_time)

print("Contiguous tensor median:", contiguous_time)

print(
    "Non-contiguous tensor is contiguous:",
    transposed_tensor.is_contiguous()
)

print(
    "Contiguous tensor is contiguous:",
    contiguous_tensor.is_contiguous()
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

os.makedirs("lab05/results", exist_ok=True)

with open(
    "lab05/results/task6_benchmark.txt",
    "w"
) as file:

    file.write("LAB 5 - TASK 6 RESULTS\n")
    file.write("======================\n\n")

    file.write(f"PyTorch version: {torch.__version__}\n")

    file.write(f"Execution mode: {execution_mode}\n")

    file.write(f"Eager median: {eager_time:.8f} seconds\n")

    file.write(
        f"Compiled/traced median: {compiled_time:.8f} seconds\n"
    )

    file.write(
        f"Non-contiguous median: {non_contiguous_time:.8f} seconds\n"
    )

    file.write(
        f"Contiguous median: {contiguous_time:.8f} seconds\n"
    )

    file.write(
        f"Non-contiguous stride: {transposed_tensor.stride()}\n"
    )

    file.write(
        f"Contiguous stride: {contiguous_tensor.stride()}\n"
    )

print("\nResults saved to lab05/results/task6_benchmark.txt")

print("\nTASK 6 EXPERIMENT COMPLETED")