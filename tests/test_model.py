from __future__ import annotations

import pytest
import torch

from waste_sort.data.dataset import CLASSES, get_train_transforms, get_val_transforms
from waste_sort.models.baseline import BaselineResNet18
from waste_sort.models.efficientnet import EfficientNetB2


class TestModels:
    @pytest.fixture
    def dummy_input(self) -> torch.Tensor:
        return torch.randn(2, 3, 224, 224)

    def test_baseline_output_shape(self, dummy_input: torch.Tensor) -> None:
        model = BaselineResNet18(num_classes=10, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (2, 10)

    def test_efficientnet_output_shape(self, dummy_input: torch.Tensor) -> None:
        model = EfficientNetB2(num_classes=10, pretrained=False, dropout=0.3)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (2, 10)

    def test_baseline_gradients(self, dummy_input: torch.Tensor) -> None:
        model = BaselineResNet18(num_classes=10, pretrained=False)
        out = model(dummy_input)
        loss = out.sum()
        loss.backward()
        assert model.backbone.fc.weight.grad is not None

    def test_efficientnet_gradients(self, dummy_input: torch.Tensor) -> None:
        model = EfficientNetB2(num_classes=10, pretrained=False)
        out = model(dummy_input)
        loss = out.sum()
        loss.backward()
        assert model.backbone.classifier[1].weight.grad is not None


class TestTransforms:
    def test_train_transforms(self) -> None:
        import numpy as np

        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        t = get_train_transforms(224)
        result = t(image=img)
        assert result["image"].shape == (3, 224, 224)

    def test_val_transforms(self) -> None:
        import numpy as np

        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        t = get_val_transforms(224)
        result = t(image=img)
        assert result["image"].shape == (3, 224, 224)


class TestClassMapping:
    def test_ten_classes(self) -> None:
        assert len(CLASSES) == 10

    def test_classes_sorted(self) -> None:
        assert CLASSES == sorted(CLASSES)
