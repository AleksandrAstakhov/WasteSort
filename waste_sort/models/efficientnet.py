
from __future__ import annotations

import torch.nn as nn
from torchvision import models
import torch


class EfficientNetB2(nn.Module):
    """EfficientNet-B2 for waste classification.

    Architecture:
        - EfficientNet-B2 backbone (ImageNet pretrained)
        - Replace classifier head -> Dropout + Linear(1408, num_classes)
        - Fine-tune all layers
    """

    def __init__(
        self,
        num_classes: int = 12,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b2(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [N, 3, H, W].

        Returns:
            Logits tensor [N, num_classes].
        """
        return self.backbone(x)
