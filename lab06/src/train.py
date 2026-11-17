"""
Lab 06: YAML-driven CNN training pipeline.

Features:
- YAML configuration
- Reproducible training
- MLflow experiment tracking
- Model checkpoint saving
- Training resume support
"""

import argparse
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

# Allow imports from the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import CNN
from data import create_dataloaders, set_seed


# Project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_device(config):
    """Select training device."""
    if config["device"]["use_cuda"] and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_optimizer(model, config):
    """Create optimizer from configuration."""
    name = config["optimizer"]["name"].lower()

    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["optimizer"].get("weight_decay", 0.0)

    if name == "adam":
        return Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "sgd":
        return SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def create_scheduler(optimizer, config):
    """Create learning-rate scheduler."""
    name = config["scheduler"]["name"].lower()
    epochs = config["training"]["epochs"]

    if name == "none":
        return None

    if name == "steplr":
        return StepLR(optimizer, step_size=3, gamma=0.1)

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)

    if name == "reducelronplateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=2,
        )

    raise ValueError(f"Unsupported scheduler: {name}")


def evaluate(model, loader, criterion, device):
    """Calculate average loss and accuracy."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_accuracy,
    history,
):
    """Save model and training state."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": best_accuracy,
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="lab06/configs/config.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="Stop after a specified epoch for interruption testing.",
    )

    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    # Reproducibility.
    seed = config["reproducibility"]["seed"]
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = get_device(config)

    print("=" * 60)
    print("LAB 06 — CNN TRAINING PIPELINE")
    print("=" * 60)
    print("Device:", device)
    print("Epochs:", config["training"]["epochs"])
    print("Batch size:", config["training"]["batch_size"])
    print("Learning rate:", config["training"]["learning_rate"])
    print("Optimizer:", config["optimizer"]["name"])

    # Dataset preparation.
    train_loader, val_loader, test_loader = create_dataloaders(config)

    print("\nDataset loaded successfully.")
    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    # Model, loss and optimizer.
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Checkpoint paths.
    checkpoint_dir = PROJECT_ROOT / config["checkpoint"]["directory"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_checkpoint = checkpoint_dir / "latest_checkpoint.pth"
    best_checkpoint = (
        checkpoint_dir / config["checkpoint"]["filename"]
    )

    start_epoch = 0
    best_accuracy = 0.0

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    # Resume training if requested.
    if args.resume:
        if not latest_checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {latest_checkpoint}"
            )

        checkpoint = torch.load(
            latest_checkpoint,
            map_location=device,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        history = checkpoint["history"]

        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])

        print(f"\nResuming from epoch {start_epoch + 1}")

    # MLflow setup.
    tracking_uri = config["logging"]["mlflow_tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = config["experiment"]["name"]
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=config["experiment"]["run_name"]
    ):

        # Log configuration parameters.
        mlflow.log_params({
            "seed": seed,
            "model": config["model"]["name"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "epochs": config["training"]["epochs"],
            "optimizer": config["optimizer"]["name"],
            "scheduler": config["scheduler"]["name"],
            "num_workers": config["training"]["num_workers"],
            "device": str(device),
        })

        print("\nStarting training...\n")

        total_epochs = config["training"]["epochs"]

        for epoch in range(start_epoch, total_epochs):

            model.train()

            running_loss = 0.0
            samples_seen = 0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                batch_count = images.size(0)

                running_loss += loss.item() * batch_count
                samples_seen += batch_count

            train_loss = running_loss / samples_seen

            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["validation_loss"].append(val_loss)
            history["validation_accuracy"].append(val_accuracy)

            epoch_number = epoch + 1

            print(
                f"Epoch {epoch_number}/{total_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Log metrics to MLflow.
            mlflow.log_metrics({
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "learning_rate": current_lr,
            }, step=epoch_number)

            # Save latest checkpoint after every epoch.
            save_checkpoint(
                latest_checkpoint,
                model,
                optimizer,
                scheduler,
                epoch_number,
                best_accuracy,
                history,
            )

            # Save best model.
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

                torch.save(
                    model.state_dict(),
                    best_checkpoint,
                )

        # Optional interruption for Task 7.
            if (
                args.stop_after_epoch is not None
                and epoch_number >= args.stop_after_epoch
            ):
                print(
                    f"\nIntentional stop after epoch {epoch_number}."
                )
                break

        # Test evaluation.
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        print("\nTraining finished.")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_validation_accuracy": best_accuracy,
        })

        # Save training history.
        history_path = PROJECT_ROOT / "lab06/results/training_history.pt"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(history, history_path)

        if config["logging"]["log_artifacts"]:
            mlflow.log_artifact(str(config_path))
            mlflow.log_artifact(str(latest_checkpoint))
            mlflow.log_artifact(str(best_checkpoint))
            mlflow.log_artifact(str(history_path))

        print("\nMLflow logging completed.")
        print("Latest checkpoint:", latest_checkpoint)
        print("Best model:", best_checkpoint)


if __name__ == "__main__":
    main()