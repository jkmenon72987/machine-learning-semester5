import os
import csv
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import OversizedCNN


# ==========================================
# 1. CONFIGURATION
# ==========================================

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.manual_seed(SEED)


# ==========================================
# 2. DATASET
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train_dataset = datasets.FashionMNIST(
    root=os.path.join(BASE_DIR, "data"),
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=os.path.join(BASE_DIR, "data"),
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

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("=" * 55)
print("FASHION-MNIST DATASET")
print("=" * 55)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Testing samples:", len(test_dataset))


# ==========================================
# 3. INITIALIZE MODEL
# ==========================================

model = OversizedCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel parameters:", total_parameters)
print("Device:", DEVICE)


# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ==========================================
# 5. TRAINING
# ==========================================

history = []

print("\n" + "=" * 55)
print("STARTING BASELINE CNN TRAINING")
print("=" * 55)

training_start = time.perf_counter()

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    epoch_start = time.perf_counter()

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_dataset)

    val_loss, val_accuracy = evaluate(model, val_loader)

    epoch_time = time.perf_counter() - epoch_start

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "epoch_time_seconds": epoch_time
    })

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy * 100:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )

training_time = time.perf_counter() - training_start


# ==========================================
# 6. TEST EVALUATION
# ==========================================

test_loss, test_accuracy = evaluate(model, test_loader)


# ==========================================
# 7. MODEL SIZE
# ==========================================

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "teacher_baseline.pth"
)

torch.save(model.state_dict(), checkpoint_path)

model_size_bytes = os.path.getsize(checkpoint_path)
model_size_mb = model_size_bytes / (1024 * 1024)


# ==========================================
# 8. INFERENCE LATENCY
# ==========================================

model.eval()

sample_input = torch.randn(1, 1, 28, 28)

# Warm-up
with torch.no_grad():

    for _ in range(10):
        model(sample_input)

iterations = 100

start_time = time.perf_counter()

with torch.no_grad():

    for _ in range(iterations):
        model(sample_input)

end_time = time.perf_counter()

latency_ms = (
    (end_time - start_time) / iterations
) * 1000


# ==========================================
# 9. SAVE RESULTS
# ==========================================

results = {
    "model": "OversizedCNN",
    "parameters": total_parameters,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "training_time_seconds": round(training_time, 2),
    "test_loss": round(test_loss, 4),
    "test_accuracy": round(test_accuracy, 4),
    "model_size_bytes": model_size_bytes,
    "model_size_mb": round(model_size_mb, 4),
    "inference_latency_ms_per_image": round(latency_ms, 4),
    "device": str(DEVICE)
}

results_path = os.path.join(
    RESULTS_DIR,
    "baseline_results.json"
)

with open(results_path, "w") as file:
    json.dump(results, file, indent=4)


history_path = os.path.join(
    RESULTS_DIR,
    "baseline_training_history.csv"
)

with open(history_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=history[0].keys()
    )

    writer.writeheader()
    writer.writerows(history)


# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("BASELINE CNN TRAINING COMPLETE")
print("=" * 55)

print("Total parameters:", total_parameters)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Model size: {model_size_mb:.4f} MB")
print(f"Inference latency: {latency_ms:.4f} ms/image")
print(f"Training time: {training_time:.2f} seconds")

print("\nSaved files:")
print("Teacher model:", checkpoint_path)
print("Results:", results_path)
print("Training history:", history_path)

print("=" * 55)