
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from waste_sort.data.dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD


def _preprocess_image(img_path: Path, image_size: int = 224) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    scale = int(image_size * 256 / 224)
    img = img.resize((scale, scale), Image.BILINEAR)
    left = (scale - image_size) // 2
    top = (scale - image_size) // 2
    img = img.crop((left, top, left + image_size, top + image_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr


class Predictor:

    def __init__(
        self,
        onnx_path: str | None = None,
        checkpoint_path: str | None = None,
        image_size: int = 224,
    ) -> None:
        self.image_size = image_size
        self.session = None
        self.model = None

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if onnx_path is not None:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        elif checkpoint_path is not None:
            from waste_sort.training.module import WasteClassifier

            self.model = WasteClassifier.load_from_checkpoint(checkpoint_path)
            self.model.eval()
            self.model = self.model.to(self.device)
        else:
            raise ValueError("Provide either onnx_path or checkpoint_path.")

    def predict_one(self, img_path: Path) -> tuple[str, float]:
        """Predict class for a single image.

        Returns:
            (class_name, confidence)
        """
        arr = _preprocess_image(img_path, self.image_size)

        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            logits = self.session.run(None, {input_name: arr})[0]
            probs = _softmax(logits[0])
        else:
            tensor = torch.from_numpy(arr).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
            probs = torch.softmax(logits[0], dim=0).cpu().numpy()

        idx = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx])

    def predict_folder(self, folder: Path, output_csv: Path) -> None:
        images = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "predicted_class", "confidence"])
            for img_path in images:
                cls_name, conf = self.predict_one(img_path)
                writer.writerow([img_path.name, cls_name, f"{conf:.4f}"])

        print(f"Predictions saved to {output_csv} ({len(images)} images)")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
