import random
import time
import statistics

import numpy as np
import torch

from sklearn.datasets import make_moons

from lab05.src.autograd_engine import Value


# ==========================================
# CONFIGURATION
# ==========================================

RANDOM_SEED = 42
SAMPLES = 200
EPOCHS = 20
LEARNING_RATE = 0.05
RUNS = 5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ==========================================
# DATASET
# ==========================================

X, y = make_moons(
    n_samples=SAMPLES,
    noise=0.2,
    random_state=RANDOM_SEED
)

X = X.astype(np.float32)
y = y.astype(np.float32)


# ==========================================
# CUSTOM NEURAL NETWORK
# Architecture: 2 -> 8 -> 1
# ==========================================

class CustomNetwork:

    def __init__(self):

        self.weights1 = [
            [Value(random.uniform(-0.5, 0.5)) for _ in range(2)]
            for _ in range(8)
        ]

        self.bias1 = [
            Value(0.0) for _ in range(8)
        ]

        self.weights2 = [
            Value(random.uniform(-0.5, 0.5))
            for _ in range(8)
        ]

        self.bias2 = Value(0.0)

        self.parameters = []

        for row in self.weights1:
            self.parameters.extend(row)

        self.parameters.extend(self.bias1)
        self.parameters.extend(self.weights2)
        self.parameters.append(self.bias2)

    def forward(self, inputs):

        hidden = []

        # Hidden layer with ReLU
        for i in range(8):

            value = self.bias1[i]

            for j in range(2):
                value = value + self.weights1[i][j] * inputs[j]

            hidden.append(value.relu())

        # Output layer
        output = self.bias2

        for i in range(8):
            output = output + self.weights2[i] * hidden[i]

        # Sigmoid output
        return output.sigmoid()


def custom_loss(prediction, target):

    epsilon = 1e-7

    # Binary cross-entropy
    return -(
        target * (prediction + epsilon).log()
        +
        (1 - target) * (1 - prediction + epsilon).log()
    )


def train_custom():

    model = CustomNetwork()

    start = time.perf_counter()

    for epoch in range(EPOCHS):

        # Reset gradients
        for parameter in model.parameters:
            parameter.grad = 0.0

        total_loss = None

        # Forward pass over all samples
        for i in range(SAMPLES):

            inputs = [
                Value(X[i][0]),
                Value(X[i][1])
            ]

            prediction = model.forward(inputs)

            loss = custom_loss(prediction, float(y[i]))

            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss

        # Average loss
        total_loss = total_loss / SAMPLES

        # Backpropagation
        total_loss.backward()

        # Update weights
        for parameter in model.parameters:

            parameter.data -= LEARNING_RATE * parameter.grad

    elapsed = time.perf_counter() - start

    return elapsed, total_loss.data


# ==========================================
# PYTORCH NEURAL NETWORK
# Architecture: 2 -> 8 -> 1
# ==========================================

class TorchNetwork(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden = torch.nn.Linear(2, 8)
        self.output = torch.nn.Linear(8, 1)

    def forward(self, x):

        x = torch.relu(self.hidden(x))

        x = torch.sigmoid(self.output(x))

        return x


def train_pytorch():

    torch.manual_seed(RANDOM_SEED)

    model = TorchNetwork()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = torch.nn.BCELoss()

    inputs = torch.tensor(X)
    targets = torch.tensor(y).reshape(-1, 1)

    start = time.perf_counter()

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        predictions = model(inputs)

        loss = criterion(predictions, targets)

        loss.backward()

        optimizer.step()

    elapsed = time.perf_counter() - start

    return elapsed, loss.item()


# ==========================================
# PERFORMANCE COMPARISON
# MEDIAN OF FIVE RUNS
# ==========================================

def main():

    custom_times = []
    torch_times = []

    print("\nLAB 5 - TASK 5")
    print("==============================")

    print("Dataset: make_moons")
    print("Samples:", SAMPLES)
    print("Architecture: 2 -> 8 -> 1")
    print("Epochs:", EPOCHS)
    print("Runs:", RUNS)

    print("\nStarting experiments...\n")

    for run in range(RUNS):

        custom_time, custom_loss_value = train_custom()

        torch_time, torch_loss_value = train_pytorch()

        custom_times.append(custom_time)
        torch_times.append(torch_time)

        print(f"Run {run + 1}")

        print(
            f"Custom engine: {custom_time:.4f} seconds"
        )

        print(
            f"PyTorch: {torch_time:.4f} seconds"
        )

        print()

    # Calculate medians
    custom_median = statistics.median(custom_times)
    torch_median = statistics.median(torch_times)

    print("==============================")
    print("FINAL PERFORMANCE RESULTS")
    print("==============================")

    print(
        f"Custom engine median: {custom_median:.4f} seconds"
    )

    print(
        f"PyTorch median: {torch_median:.4f} seconds"
    )

    if custom_median > torch_median:

        ratio = custom_median / torch_median

        print(
            f"PyTorch was approximately {ratio:.2f}x faster."
        )

    else:

        ratio = torch_median / custom_median

        print(
            f"Custom engine was approximately {ratio:.2f}x faster."
        )

    print("\nFinal custom loss:", custom_loss_value)
    print("Final PyTorch loss:", torch_loss_value)

    # Save benchmark results
    import os

    os.makedirs("lab05/results", exist_ok=True)

    with open(
        "lab05/results/task5_benchmark.txt",
        "w"
    ) as file:

        file.write("LAB 5 - TASK 5 RESULTS\n")
        file.write("======================\n")

        file.write(f"Dataset samples: {SAMPLES}\n")
        file.write(f"Epochs: {EPOCHS}\n")
        file.write(f"Runs: {RUNS}\n")

        file.write(
            f"Custom median: {custom_median:.6f} seconds\n"
        )

        file.write(
            f"PyTorch median: {torch_median:.6f} seconds\n"
        )

        file.write(
            f"Custom loss: {custom_loss_value}\n"
        )

        file.write(
            f"PyTorch loss: {torch_loss_value}\n"
        )

    print("\nResults saved to lab05/results/task5_benchmark.txt")


if __name__ == "__main__":
    main()