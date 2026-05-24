from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from waste_sort.data.dataset import CLASSES
from waste_sort.models.baseline import BaselineResNet18
from waste_sort.models.efficientnet import EfficientNetB2

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline": BaselineResNet18,
    "efficientnet": EfficientNetB2,
}


class WasteClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "efficientnet",
        num_classes: int = 12,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        label_smoothing: float = 0.1,
        scheduler_t_max: int = 50,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_cls = MODEL_REGISTRY[model_name]
        if model_name == "efficientnet":
            self.model = model_cls(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
        else:
            self.model = model_cls(num_classes=num_classes, pretrained=pretrained)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None)
        self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = batch["image"]
        labels = batch["label"]
        logits = self(images)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, labels = self._shared_step(batch)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, labels)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/accuracy", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        loss, logits, labels = self._shared_step(batch)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/accuracy", self.val_acc, on_epoch=True)
        self.log("val/f1_macro", self.val_f1, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        loss, logits, labels = self._shared_step(batch)
        preds = logits.argmax(dim=1)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_f1_per_class(preds, labels)
        self.test_cm(preds, labels)
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/accuracy", self.test_acc, on_epoch=True)
        self.log("test/f1_macro", self.test_f1, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        per_class_f1 = self.test_f1_per_class.compute()
        for i, class_name in enumerate(CLASSES):
            self.log(f"test/f1_{class_name}", per_class_f1[i])

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.scheduler_t_max,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
