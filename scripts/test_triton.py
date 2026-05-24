#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import requests
from PIL import Image

from waste_sort.data.dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD


def preprocess_image(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    scale = int(224 * 256 / 224)
    img = img.resize((scale, scale), Image.BILINEAR)
    left = (scale - 224) // 2
    top = (scale - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr


def test_triton():
    print("Testing Triton Inference (HTTP API):\n")

    test_images = [
        ("data/raw/biological/biological1.jpg", "biological"),
        ("data/raw/paper/paper1.jpg", "paper"),
        ("data/raw/plastic/plastic1.jpg", "plastic"),
    ]

    correct = 0
    for img_path_str, expected in test_images:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"{img_path.name}: file not found")
            continue

        img_array = preprocess_image(img_path)

        request = {
            "inputs": [
                {
                    "name": "input",
                    "shape": list(img_array.shape),
                    "datatype": "FP32",
                    "data": img_array.flatten().tolist(),
                }
            ]
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/v2/models/waste_sort/infer",
                json=request,
                timeout=5,
            )

            if response.status_code == 200:
                result = response.json()
                logits = np.array(result["outputs"][0]["data"])
                if len(logits.shape) > 1:
                    logits = logits.flatten()

                predicted_idx = np.argmax(logits[: len(CLASSES)])
                predicted_class = CLASSES[predicted_idx]
                confidence = float(
                    np.exp(logits[predicted_idx]) / np.exp(logits[: len(CLASSES)]).sum()
                )

                is_correct = predicted_class == expected
                status = "ок" if is_correct else "not ok"
                print(f"{status} {img_path.name:25}  {predicted_class:12} ({confidence:.4f})")
                if is_correct:
                    correct += 1
            else:
                print(f"{img_path.name}: HTTP {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("Connection failed. Make sure Triton is running:")
            print("   tritonserver --model-repository=$(pwd)/triton_repo")
            return
        except Exception as e:
            print(f"{img_path.name}: {e}")

    print(f"\nAccuracy: {correct}/{len(test_images)}")


if __name__ == "__main__":
    test_triton()
