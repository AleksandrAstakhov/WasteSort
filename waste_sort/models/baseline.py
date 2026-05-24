from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class BaselineResNet18(nn.Module):
    """ResNet-18 fine-tuned for waste classification.

    Architecture:
        - ResNet-18 backbone (ImageNet pretrained)
        - Replace final FC -> nn.Linear(512, num_classes)
        - Fine-tune all layers
    """

    def __init__(self, num_classes: int = 12, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [N, 3, H, W].

        Returns:
            Logits tensor [N, num_classes].
        """
        return self.backbone(x)
