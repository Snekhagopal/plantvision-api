
import torch
from torch import nn
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)


def create_mobilenet(num_classes=38, pretrained=True):
    # Load MobileNetV2 with pretrained ImageNet weights. Use
    # pretrained=False when a full checkpoint will be loaded afterwards,
    # so inference never needs to download ImageNet weights.
    model = mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT if pretrained else None
    )

    # Freeze pretrained feature extraction layers.
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the original classifier with our 38 plant classes.
    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        num_classes,
    )

    return model


if __name__ == "__main__":
    model = create_mobilenet(num_classes=38)
    dummy_input = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters()
            if p.requires_grad),
    )
    print("MobileNetV2 model created successfully!")
