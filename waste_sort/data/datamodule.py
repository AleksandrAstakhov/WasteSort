from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from waste_sort.data.dataset import CLASSES, WasteDataset, get_train_transforms, get_val_transforms


class WasteDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/raw",
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
        split_file: str = "artifacts/split.json",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.split_file = Path(split_file)

        self.train_dataset: WasteDataset | None = None
        self.val_dataset: WasteDataset | None = None
        self.test_dataset: WasteDataset | None = None

    def _collect_samples(self) -> tuple[list[Path], list[int]]:
        paths: list[Path] = []
        labels: list[int] = []
        class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}

        for class_name in CLASSES:
            class_dir = self.data_dir / class_name
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    paths.append(img_path)
                    labels.append(class_to_idx[class_name])

        return paths, labels

    def setup(self, stage: str | None = None) -> None:
        paths, labels = self._collect_samples()

        if len(paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.data_dir}. "
                "Run `waste-sort download` or `dvc repro get_data` first."
            )

        test_ratio = 1.0 - self.train_ratio - self.val_ratio
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths,
            labels,
            test_size=(self.val_ratio + test_ratio),
            stratify=labels,
            random_state=self.seed,
        )
        relative_val = self.val_ratio / (self.val_ratio + test_ratio)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            test_size=(1 - relative_val),
            stratify=temp_labels,
            random_state=self.seed,
        )

        # Save split indices for reproducibility
        self.split_file.parent.mkdir(parents=True, exist_ok=True)
        split_info = {
            "train": [str(p) for p in train_paths],
            "val": [str(p) for p in val_paths],
            "test": [str(p) for p in test_paths],
            "seed": self.seed,
        }
        self.split_file.write_text(json.dumps(split_info, indent=2))

        train_transform = get_train_transforms(self.image_size)
        val_transform = get_val_transforms(self.image_size)

        self.train_dataset = WasteDataset(train_paths, train_labels, train_transform)
        self.val_dataset = WasteDataset(val_paths, val_labels, val_transform)
        self.test_dataset = WasteDataset(test_paths, test_labels, val_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
