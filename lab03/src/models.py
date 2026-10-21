
"""
Lab 03: MLP, CNN, and Transformer architectures for Fashion-MNIST.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected neural network."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    """Convolutional neural network."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class AttentionModel(nn.Module):
    """Patch-based Transformer encoder for image classification."""

    def __init__(self):
        super().__init__()

        self.patch_projection = nn.Linear(7 * 7, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
        batch_size = x.shape[0]

        # Convert each image into sixteen 7x7 patches.
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.contiguous().view(batch_size, 16, 49)

        # Project each patch into a 64-dimensional vector.
        x = self.patch_projection(patches)

        # Apply two Transformer encoder layers.
        x = self.transformer(x)

        # Average the patch representations.
        x = x.mean(dim=1)

        # Produce ten class scores.
        return self.classifier(x)


if __name__ == "__main__":
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    for name, model in models.items():
        total_parameters = sum(
            parameter.numel() for parameter in model.parameters()
        )

        with torch.no_grad():
            sample = torch.randn(2, 1, 28, 28)
            output = model(sample)

        print(f"\n{name}")
        print(f"Parameters: {total_parameters:,}")
        print(f"Output shape: {tuple(output.shape)}")

