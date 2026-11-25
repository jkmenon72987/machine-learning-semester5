import os
import csv
import json
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import OversizedCNN, StudentCNN


# ==========================================
# 1. CONFIGURATION
# ==========================================

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

TEMPERATURE = 4
ALPHA = 0.5

DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==========================================
# 2. REPRODUCIBILITY
# ==========================================

torch.manual_seed(SEED)


# ==========================================
# 3. LOAD FASHION-MNIST
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

print("=" * 60)
print("KNOWLEDGE DISTILLATION EXPERIMENT")
print("=" * 60)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Testing samples:", len(test_dataset))


# ==========================================
# 4. LOAD TRAINED TEACHER
# ==========================================

teacher = OversizedCNN().to(DEVICE)

teacher_path = os.path.join(
    CHECKPOINT_DIR,
    "teacher_baseline.pth"
)

teacher.load_state_dict(
    torch.load(teacher_path, map_location=DEVICE)
)

teacher.eval()

for parameter in teacher.parameters():
    parameter.requires_grad = False

teacher_parameters = sum(
    p.numel() for p in teacher.parameters()
)

print("\nTeacher parameters:", teacher_parameters)


# ==========================================
# 5. LOSS FUNCTIONS
# ==========================================

criterion_ce = nn.CrossEntropyLoss()

criterion_kl = nn.KLDivLoss(
    reduction="batchmean"
)


# ==========================================
# 6. EVALUATION
# ==========================================

def evaluate(model):

    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion_ce(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    average_loss = total_loss / total

    return average_loss, accuracy


# ==========================================
# 7. TRAIN STUDENT WITHOUT DISTILLATION
# ==========================================

def train_student_normal():

    print("\n" + "=" * 60)
    print("STUDENT TRAINING WITHOUT DISTILLATION")
    print("=" * 60)

    torch.manual_seed(SEED)

    model = StudentCNN().to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion_ce(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_dataset)

        _, val_accuracy = evaluate(model)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Test Accuracy: {val_accuracy * 100:.2f}%"
        )

    training_time = time.perf_counter() - start_time

    return model, training_time


# ==========================================
# 8. TRAIN STUDENT WITH DISTILLATION
# ==========================================

def train_student_distillation():

    print("\n" + "=" * 60)
    print("STUDENT TRAINING WITH KNOWLEDGE DISTILLATION")
    print("=" * 60)

    torch.manual_seed(SEED)

    model = StudentCNN().to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        running_ce = 0.0
        running_kl = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Student predictions
            student_outputs = model(images)

            # Teacher predictions
            with torch.no_grad():

                teacher_outputs = teacher(images)

            # Classification loss
            ce_loss = criterion_ce(
                student_outputs,
                labels
            )

            # Soft probability distributions
            student_log_probs = nn.functional.log_softmax(
                student_outputs / TEMPERATURE,
                dim=1
            )

            teacher_probs = nn.functional.softmax(
                teacher_outputs / TEMPERATURE,
                dim=1
            )

            # Knowledge distillation loss
            kl_loss = criterion_kl(
                student_log_probs,
                teacher_probs
            ) * (TEMPERATURE ** 2)

            # Combined loss
            loss = (
                ALPHA * ce_loss
                + (1 - ALPHA) * kl_loss
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_ce += ce_loss.item() * labels.size(0)
            running_kl += kl_loss.item() * labels.size(0)

        train_loss = running_loss / len(train_dataset)
        average_ce = running_ce / len(train_dataset)
        average_kl = running_kl / len(train_dataset)

        _, val_accuracy = evaluate(model)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Total Loss: {train_loss:.4f} | "
            f"CE: {average_ce:.4f} | "
            f"KL: {average_kl:.4f} | "
            f"Test Accuracy: {val_accuracy * 100:.2f}%"
        )

    training_time = time.perf_counter() - start_time

    return model, training_time


# ==========================================
# 9. MODEL SIZE
# ==========================================

def save_model(model, filename):

    path = os.path.join(
        CHECKPOINT_DIR,
        filename
    )

    torch.save(model.state_dict(), path)

    size_mb = os.path.getsize(path) / (1024 * 1024)

    return path, size_mb


# ==========================================
# 10. INFERENCE LATENCY
# ==========================================

def measure_latency(model, iterations=200):

    model.eval()

    sample = torch.randn(1, 1, 28, 28)

    with torch.no_grad():

        for _ in range(20):
            model(sample)

    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(iterations):
            model(sample)

    end = time.perf_counter()

    return ((end - start) / iterations) * 1000


# ==========================================
# 11. RUN BOTH EXPERIMENTS
# ==========================================

normal_student, normal_training_time = train_student_normal()

distilled_student, distilled_training_time = train_student_distillation()


# ==========================================
# 12. EVALUATE FINAL MODELS
# ==========================================

normal_loss, normal_accuracy = evaluate(normal_student)

distilled_loss, distilled_accuracy = evaluate(distilled_student)

teacher_loss, teacher_accuracy = evaluate(teacher)


# ==========================================
# 13. SAVE CHECKPOINTS
# ==========================================

normal_path, normal_size = save_model(
    normal_student,
    "student_normal.pth"
)

distilled_path, distilled_size = save_model(
    distilled_student,
    "student_distilled.pth"
)


# ==========================================
# 14. LATENCY
# ==========================================

normal_latency = measure_latency(normal_student)

distilled_latency = measure_latency(distilled_student)

teacher_latency = measure_latency(teacher)


# ==========================================
# 15. SAVE COMPARISON
# ==========================================

student_parameters = sum(
    p.numel() for p in normal_student.parameters()
)

results = [

    {
        "model": "Teacher CNN",
        "parameters": teacher_parameters,
        "test_loss": teacher_loss,
        "test_accuracy": teacher_accuracy,
        "model_size_mb": os.path.getsize(teacher_path) / (1024 * 1024),
        "latency_ms": teacher_latency,
        "training_time_seconds": None
    },

    {
        "model": "Student Normal",
        "parameters": student_parameters,
        "test_loss": normal_loss,
        "test_accuracy": normal_accuracy,
        "model_size_mb": normal_size,
        "latency_ms": normal_latency,
        "training_time_seconds": normal_training_time
    },

    {
        "model": "Student Distilled",
        "parameters": student_parameters,
        "test_loss": distilled_loss,
        "test_accuracy": distilled_accuracy,
        "model_size_mb": distilled_size,
        "latency_ms": distilled_latency,
        "training_time_seconds": distilled_training_time
    }
]


# Save CSV
csv_path = os.path.join(
    RESULTS_DIR,
    "distillation_comparison.csv"
)

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# Save JSON
json_path = os.path.join(
    RESULTS_DIR,
    "distillation_comparison.json"
)

with open(json_path, "w") as file:

    json.dump(results, file, indent=4)


# ==========================================
# 16. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 65)
print("KNOWLEDGE DISTILLATION COMPLETE")
print("=" * 65)

for result in results:

    print(
        f"\nModel: {result['model']}"
    )

    print(
        f"Parameters: {result['parameters']}"
    )

    print(
        f"Test accuracy: {result['test_accuracy'] * 100:.2f}%"
    )

    print(
        f"Model size: {result['model_size_mb']:.4f} MB"
    )

    print(
        f"Latency: {result['latency_ms']:.4f} ms/image"
    )

print("\nResults saved:")
print(csv_path)
print(json_path)

print("\nStudent checkpoints saved:")
print(normal_path)
print(distilled_path)

print("=" * 65)