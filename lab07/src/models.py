import torch
import torch.nn as nn


# ==========================================
# TEACHER MODEL - OVERSIZED CNN
# ==========================================

class OversizedCNN(nn.Module):

    def __init__(self):
        super(OversizedCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 448),
            nn.ReLU(),

            nn.Linear(448, 10)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ==========================================
# STUDENT MODEL - COMPACT CNN
# ==========================================

class StudentCNN(nn.Module):

    def __init__(self):
        super(StudentCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ==========================================
# MODEL VERIFICATION
# ==========================================

if __name__ == "__main__":

    teacher = OversizedCNN()
    student = StudentCNN()

    teacher_params = sum(
        p.numel() for p in teacher.parameters()
    )

    student_params = sum(
        p.numel() for p in student.parameters()
    )

    print("TEACHER MODEL")
    print("Parameters:", teacher_params)

    print("\nSTUDENT MODEL")
    print("Parameters:", student_params)

    reduction = (
        1 - student_params / teacher_params
    ) * 100

    print(f"\nParameter reduction: {reduction:.2f}%")