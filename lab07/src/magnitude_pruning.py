import os
import csv
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import OversizedCNN


# ==========================================
# 1. CONFIGURATION
# ==========================================

SEED = 42
BATCH_SIZE = 128
FINETUNE_EPOCHS = 3
LEARNING_RATE = 0.0005

SPARSITY_LEVELS = [0, 0.25, 0.50, 0.75, 0.90, 0.95]

DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.manual_seed(SEED)


# ==========================================
# 2. LOAD DATASET
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 3. LOAD BASELINE MODEL
# ==========================================

BASELINE_PATH = os.path.join(
    CHECKPOINT_DIR,
    "teacher_baseline.pth"
)

baseline_state = torch.load(
    BASELINE_PATH,
    map_location=DEVICE
)

criterion = nn.CrossEntropyLoss()


# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================

def evaluate(model):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ==========================================
# 5. IDENTIFY PRUNABLE LAYERS
# ==========================================

def get_prunable_layers(model):

    return [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
        or isinstance(module, nn.Linear)
    ]


# ==========================================
# 6. COUNT ACTUAL ZERO WEIGHTS
# ==========================================

def calculate_sparsity(model):

    total_weights = 0
    zero_weights = 0

    for module in model.modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weights = module.weight.detach()

            total_weights += weights.numel()
            zero_weights += (weights == 0).sum().item()

    sparsity = zero_weights / total_weights

    return sparsity


# ==========================================
# 7. APPLY MAGNITUDE PRUNING
# ==========================================

def apply_pruning(model, method, amount):

    layers = get_prunable_layers(model)

    if amount == 0:
        return

    if method == "global":

        prune.global_unstructured(
            layers,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )

    elif method == "layerwise":

        for module, parameter_name in layers:

            prune.l1_unstructured(
                module,
                name=parameter_name,
                amount=amount
            )


# ==========================================
# 8. FINE-TUNING
# ==========================================

def fine_tune(model):

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(FINETUNE_EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        print(
            f"    Fine-tune Epoch "
            f"{epoch + 1}/{FINETUNE_EPOCHS} "
            f"| Loss: {average_loss:.4f}"
        )


# ==========================================
# 9. RUN EXPERIMENTS
# ==========================================

results = []

methods = ["global", "layerwise"]

for method in methods:

    print("\n" + "=" * 65)
    print(f"STARTING {method.upper()} MAGNITUDE PRUNING")
    print("=" * 65)

    for sparsity_target in SPARSITY_LEVELS:

        print(
            f"\nTarget sparsity: "
            f"{sparsity_target * 100:.0f}%"
        )

        model = OversizedCNN().to(DEVICE)

        model.load_state_dict(
            copy.deepcopy(baseline_state)
        )

        apply_pruning(
            model,
            method,
            sparsity_target
        )

        actual_sparsity_before = calculate_sparsity(model)

        print(
            f"Actual sparsity before fine-tuning: "
            f"{actual_sparsity_before * 100:.2f}%"
        )

        if sparsity_target > 0:

            fine_tune(model)

        # Pruning masks remain active during fine-tuning.
        # Remove reparameterization after fine-tuning.
        for module, parameter_name in get_prunable_layers(model):

            if hasattr(module, parameter_name + "_orig"):

                prune.remove(module, parameter_name)

        final_sparsity = calculate_sparsity(model)

        test_accuracy = evaluate(model)

        result = {
            "method": method,
            "target_sparsity": sparsity_target,
            "actual_sparsity": final_sparsity,
            "test_accuracy": test_accuracy,
            "accuracy_percent": test_accuracy * 100
        }

        results.append(result)

        print("\nRESULT")
        print("Method:", method)
        print("Target sparsity:", sparsity_target)
        print(f"Actual sparsity: {final_sparsity * 100:.2f}%")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")

        checkpoint_name = (
            f"{method}_pruning_"
            f"{int(sparsity_target * 100)}.pth"
        )

        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, checkpoint_name)
        )


# ==========================================
# 10. SAVE RESULTS CSV
# ==========================================

csv_path = os.path.join(
    RESULTS_DIR,
    "magnitude_pruning_results.csv"
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

print("\n" + "=" * 65)
print("MAGNITUDE PRUNING EXPERIMENTS COMPLETE")
print("=" * 65)

for result in results:

    print(
        f"{result['method']:10s} | "
        f"Target: {result['target_sparsity'] * 100:5.1f}% | "
        f"Actual: {result['actual_sparsity'] * 100:5.1f}% | "
        f"Accuracy: {result['accuracy_percent']:.2f}%"
    )

print("\nResults saved to:", csv_path)