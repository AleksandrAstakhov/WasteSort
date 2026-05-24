#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from waste_sort.inference.predictor import Predictor


def test_pytorch_inference():
    print("Testing PyTorch Inference...\n")

    predictor = Predictor(
        checkpoint_path="artifacts/checkpoints/efficientnet-best.ckpt"
    )

    test_img = Path("data/raw/paper/paper1.jpg")
    if test_img.exists():
        class_name, confidence = predictor.predict_one(test_img)
        print(f"Prediction: {class_name} (confidence: {confidence:.4f})")
    else:
        print("Test image not found")


def test_onnx_inference():
    print("\nTesting ONNX Inference...\n")

    predictor = Predictor(onnx_path="artifacts/model.onnx")

    test_images = [
        ("data/raw/biological/biological1.jpg", "biological"),
        ("data/raw/paper/paper1.jpg", "paper"),
        ("data/raw/plastic/plastic1.jpg", "plastic"),
    ]

    correct = 0
    for img_path_str, expected in test_images:
        img_path = Path(img_path_str)
        if img_path.exists():
            class_name, confidence = predictor.predict_one(img_path)
            if class_name == expected:
                print(f"{img_path.name:25}  {class_name:12} ({confidence:.4f})")
                correct += 1
            else:
                print(f"{img_path.name:25}  expected {expected}, got {class_name}")
        else:
            print(f"{img_path.name:25}  file not found")

    print(f"\nAccuracy: {correct}/{len(test_images)}")


if __name__ == "__main__":
    test_pytorch_inference()
    test_onnx_inference()
