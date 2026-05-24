#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_data():
    print("\nDATA VERIFICATION")

    data_root = Path("data/raw")
    if not data_root.exists():
        print("data/raw not found")
        return False

    classes = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    total_images = sum(len(list((data_root / c).glob("*.jpg"))) for c in classes)

    print(f"Dataset: {len(classes)} classes, {total_images} images")

    if Path("artifacts/class_map.json").exists():
        with open("artifacts/class_map.json") as f:
            class_map = json.load(f)
            print(f"Class map: {len(class_map)} classes")
    else:
        print("class_map.json not found")

    if Path("artifacts/split.json").exists():
        with open("artifacts/split.json") as f:
            split = json.load(f)
            print(
                f"Train/Val/Test: {len(split['train'])}/{len(split['val'])}/{len(split['test'])}"
            )
    else:
        print("split.json not found")

    return True


def check_checkpoints():
    print("\nMODEL CHECKPOINTS")

    import torch

    models = [
        "artifacts/checkpoints/baseline-best.ckpt",
        "artifacts/checkpoints/efficientnet-best.ckpt",
    ]

    found = 0
    for model_path in models:
        path = Path(model_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            ckpt = torch.load(path, map_location="cpu")
            num_classes = ckpt["hyper_parameters"].get("num_classes", "?")
            print(f"{path.name}: {size_mb:.0f}MB")
            found += 1
        else:
            print(f"{path.name} not found")

    return found > 0


def check_onnx():
    print("\nONNX MODELS")

    onnx_models = ["artifacts/model.onnx", "artifacts/baseline_model.onnx"]

    found = 0
    for onnx_path in onnx_models:
        path = Path(onnx_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"{path.name}: {size_mb:.2f}MB")
            found += 1
        else:
            print(f"{path.name} not found")

    return found > 0


def check_inference():
    print("\nINFERENCE TESTS")

    try:
        from waste_sort.inference.predictor import Predictor

        # Test ONNX
        predictor = Predictor(onnx_path="artifacts/model.onnx")
        test_img = Path("data/raw/paper/paper1.jpg")
        if test_img.exists():
            class_name, confidence = predictor.predict_one(test_img)
            print(f"ONNX inference: {class_name} ({confidence:.4f})")

        # Test PyTorch
        predictor = Predictor(
            checkpoint_path="artifacts/checkpoints/efficientnet-best-v2.ckpt"
        )
        class_name, confidence = predictor.predict_one(test_img)
        print(f"PyTorch inference: {class_name} ({confidence:.4f})")

        return True
    except Exception as e:
        print(f"Inference failed: {e}")
        return False


def check_triton():
    print("\nTRITON DEPLOYMENT")

    try:
        result = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:8000/v2/health/ready"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"Triton server: Running")
            return True
        else:
            print(f"Triton server: Not running")
            print(
                f"Start with: tritonserver --model-repository=$(pwd)/triton_repo"
            )
            return False
    except Exception as e:
        print(f"Triton check failed: {e}")
        return False


def check_mlflow():
    print("\nMLFLOW TRACKING")

    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        runs = mlflow.search_runs(experiment_names=["waste-sort"], max_results=3)
        print(f"MLflow: {len(runs)} training runs")
        return True
    except Exception as e:
        print(f"MLflow: {e}")
        print(f"Start with: mlflow server --host 127.0.0.1 --port 8080")
        return False


def main():
    print("\nSYSTEM CHECK\n" + "=" * 60 + "\n")

    checks = [
        check_data,
        check_checkpoints,
        check_onnx,
        check_inference,
        check_triton,
        check_mlflow,
    ]

    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"Check failed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    if all(results[:3]):
        print("System is ready!")
    else:
        print("Some components missing")


if __name__ == "__main__":
    main()
