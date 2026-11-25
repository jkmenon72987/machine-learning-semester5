import os
import csv
import copy
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
EPOCHS = 3
LEARNING_RATE = 0.0005

PRUNING_LEVELS = [0.10, 0.25, 0.50]

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
# 3. LOAD BASELINE CHECKPOINT
# ==========================================

baseline_path = os.path.join(
    CHECKPOINT_DIR,
    "teacher_baseline.pth"
)

baseline_state = torch.load(
    baseline_path,
    map_location=DEVICE
)

criterion = nn.CrossEntropyLoss()


# ==========================================
# 4. GET CONVOLUTIONAL LAYERS
# ==========================================

def get_conv_layers(model):

    return [
        module
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]


# ==========================================
# 5. CALCULATE FILTER SPARSITY
# ==========================================

def calculate_filter_sparsity(model):

    total_filters = 0
    zero_filters = 0

    for layer in get_conv_layers(model):

        weights = layer.weight.detach()

        filter_norms = torch.norm(
            weights.reshape(weights.size(0), -1),
            p=2,
            dim=1
        )

        total_filters += filter_norms.numel()
        zero_filters += (filter_norms == 0).sum().item()

    return zero_filters / total_filters


# ==========================================
# 6. EVALUATE MODEL
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
# 7. FINE-TUNING
# ==========================================

def fine_tune(model):

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Fine-tune Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {average_loss:.4f}"
        )


# ==========================================
# 8. STRUCTURED PRUNING EXPERIMENT
# ==========================================

results = []

for amount in PRUNING_LEVELS:

    print("\n" + "=" * 60)
    print(f"STRUCTURED PRUNING: {amount * 100:.0f}%")
    print("=" * 60)

    model = OversizedCNN().to(DEVICE)

    model.load_state_dict(
        copy.deepcopy(baseline_state)
    )

    # Prune complete convolutional filters
    for layer in get_conv_layers(model):

        prune.ln_structured(
            layer,
            name="weight",
            amount=amount,
            n=2,
            dim=0
        )

    initial_filter_sparsity = calculate_filter_sparsity(model)

    print(
        f"Filter sparsity before fine-tuning: "
        f"{initial_filter_sparsity * 100:.2f}%"
    )

    # Masks remain fixed during fine-tuning
    fine_tune(model)

    # Make pruning permanent
    for layer in get_conv_layers(model):

        if hasattr(layer, "weight_orig"):
            prune.remove(layer, "weight")

    final_filter_sparsity = calculate_filter_sparsity(model)

    accuracy = evaluate(model)

    result = {
        "method": "structured_l2_filter",
        "target_sparsity": amount,
        "actual_filter_sparsity": final_filter_sparsity,
        "test_accuracy": accuracy,
        "accuracy_percent": accuracy * 100
    }

    results.append(result)

    print("\nRESULT")
    print(f"Target filter pruning: {amount * 100:.0f}%")
    print(
        f"Actual filter sparsity: "
        f"{final_filter_sparsity * 100:.2f}%"
    )
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    checkpoint_name = (
        f"structured_pruning_{int(amount * 100)}.pth"
    )

    torch.save(
        model.state_dict(),
        os.path.join(CHECKPOINT_DIR, checkpoint_name)
    )


# ==========================================
# 9. SAVE RESULTS
# ==========================================

csv_path = os.path.join(
    RESULTS_DIR,
    "structured_pruning_results.csv"
)

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# 10. SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("STRUCTURED PRUNING COMPLETE")
print("=" * 60)

for result in results:

    print(
        f"Target: {result['target_sparsity'] * 100:.0f}% | "
        f"Actual: {result['actual_filter_sparsity'] * 100:.2f}% | "
        f"Accuracy: {result['accuracy_percent']:.2f}%"
    )

print("\nResults saved:", csv_path)