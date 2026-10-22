
import torch
from torch import nn


class PlantCNN(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: detect simple patterns
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: learn more complex patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: learn higher-level features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Reduce spatial dimensions
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = PlantCNN(num_classes=38)
    dummy_input = torch.randn(2, 3, 256, 256)

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Model created successfully!")
    print("Trainable parameters:",
          sum(p.numel() for p in model.parameters()
              if p.requires_grad))
