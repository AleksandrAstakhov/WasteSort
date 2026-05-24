from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

CLASSES = [
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metals",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "unknown",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    resize_size = int(image_size * 256 / 224)
    return A.Compose(
        [
            A.Resize(height=resize_size, width=resize_size),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class WasteDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: A.Compose | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_np = np.array(img)

        if self.transform is not None:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            img_tensor = ToTensorV2()(image=img_np)["image"]

        return {"image": img_tensor, "label": self.labels[idx]}

    @staticmethod
    def save_class_map(path: Path) -> None:
        mapping = {name: idx for idx, name in enumerate(CLASSES)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping, indent=2))

    @staticmethod
    def load_class_map(path: Path) -> dict[str, int]:
        return json.loads(path.read_text())
