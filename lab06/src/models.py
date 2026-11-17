"""
Lab 06: CNN model for Fashion-MNIST.

Architecture reused from Lab 03.
"""

import torch.nn as nn


class CNN(nn.Module):
    """Convolutional neural network for Fashion-MNIST classification."""

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


if __name__ == "__main__":
    import torch

    model = CNN()
    sample = torch.randn(2, 1, 28, 28)
    output = model(sample)

    parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("Lab 06 CNN model")
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(output.shape))
    print("Parameters:", f"{parameters:,}")