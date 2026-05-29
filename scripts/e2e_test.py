#!/usr/bin/env python3

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002
TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:24.07-py3"
TRITON_CONTAINER = "waste_sort_triton"


def step(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def ok(msg: str) -> None:
    print(f"  ok  {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    sys.exit(1)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, cwd=ROOT, **kwargs)


def poetry(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return run(["poetry", "run", *args], **kwargs)


def run_unit_tests() -> None:
    step("1 / 7  Unit tests (pytest)")
    run(["poetry", "run", "pytest", "tests/", "-v", "--tb=short"], capture_output=False)
    ok("All unit tests passed")


def ensure_data() -> None:
    step("2 / 7  Data")

    data_dir = ROOT / "data" / "raw"
    if data_dir.exists() and len(list(data_dir.iterdir())) >= 10:
        ok(f"Data already present: {data_dir}")
        return

    dvc_storage = ROOT.parent / "dvc-storage" / "data"
    if not dvc_storage.exists():
        print(f"  Setting up DVC remote at {dvc_storage}...")
        dvc_storage.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["poetry", "run", "dvc", "remote", "add", "-d", "data-storage", str(dvc_storage)],
            cwd=ROOT, capture_output=True,
        )

    print("  Trying dvc pull...")
    result = subprocess.run(
        ["poetry", "run", "dvc", "pull"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode == 0 and data_dir.exists() and len(list(data_dir.iterdir())) >= 10:
        ok("Data pulled via DVC")
        return

    print("  DVC pull failed, downloading from Kaggle...")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists() and not os.environ.get("KAGGLE_API_TOKEN"):
        fail(
            "No Kaggle credentials found.\n"
            "  Set KAGGLE_API_TOKEN=username:api_key or create ~/.kaggle/kaggle.json"
        )

    poetry("python", "scripts/download_data.py")
    ok("Data downloaded from Kaggle")

    subprocess.run(["poetry", "run", "dvc", "add", "data/raw/"], cwd=ROOT, check=False)
    subprocess.run(["poetry", "run", "dvc", "push", "-r", "data-storage"], cwd=ROOT, check=False)
    ok("Data tracked with DVC")


def check_mlflow() -> None:
    step("3 / 7  MLflow check")

    result = poetry(
        "python", "-c",
        "import mlflow; print(mlflow.__version__)",
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        fail(f"MLflow not importable: {result.stderr.strip()}")
    version = result.stdout.strip()
    if int(version.split(".")[0]) < 3:
        fail(f"MLflow >= 3.0 required, found {version}")
    ok(f"MLflow {version}")

    db_uri = f"sqlite:///{ROOT}/mlruns.db"
    result = poetry(
        "python", "-c",
        f"import mlflow; mlflow.set_tracking_uri('{db_uri}'); "
        "mlflow.set_experiment('e2e_check'); "
        "run = mlflow.start_run(); mlflow.log_metric('ok', 1); mlflow.end_run(); "
        "print('tracking ok')",
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        fail(f"MLflow tracking failed: {result.stderr.strip()}")
    ok(result.stdout.strip())


def train_one_epoch() -> None:
    step("4 / 7  Training (1 epoch)")

    (ROOT / "artifacts" / "checkpoints").mkdir(parents=True, exist_ok=True)

    db_uri = f"sqlite:///{ROOT}/mlruns.db"
    poetry(
        "python", "scripts/run_training_patched.py",
        "++command=train",
        "model=efficientnet",
        "train.max_epochs=1",
        "train.patience=1",
        f"mlflow.tracking_uri={db_uri}",
    )

    ckpt_dir = ROOT / "artifacts" / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        fail(f"No checkpoint found in {ckpt_dir}")
    ok(f"Checkpoint saved: {max(ckpts, key=lambda p: p.stat().st_mtime).name}")

    print("  Running check_system.py...")
    poetry("python", "scripts/check_system.py")


def export_and_infer_onnx() -> None:
    step("5 / 7  ONNX export + inference")

    poetry("python", "scripts/export_onnx.py")

    onnx_path = ROOT / "artifacts" / "model.onnx"
    if not onnx_path.exists():
        fail(f"ONNX model not found: {onnx_path}")
    ok(f"ONNX exported: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")

    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from waste_sort.data.dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    test_dir = ROOT / "data" / "raw"
    tested = 0
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        imgs = list(class_dir.glob("*.jpg"))
        if not imgs:
            continue
        img = Image.open(imgs[0]).convert("RGB").resize((224, 224))
        arr = (np.array(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        arr = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        logits = sess.run(None, {input_name: arr})[0][0]
        pred = CLASSES[int(np.argmax(logits))]
        print(f"    {imgs[0].parent.name:12} -> {pred}")
        tested += 1
        if tested >= 3:
            break

    ok(f"ONNX inference OK ({tested} samples)")


def build_triton_repo() -> None:
    step("6 / 7  Triton model repository")

    from waste_sort.serving.triton import build_triton_repo as _build

    repo = _build(
        onnx_path=str(ROOT / "artifacts" / "model.onnx"),
        repo_path=str(ROOT / "triton_repo"),
        model_name="waste_sort",
        max_batch_size=8,
        image_size=224,
        num_classes=10,
    )

    config = repo / "waste_sort" / "config.pbtxt"
    model_file = repo / "waste_sort" / "1" / "model.onnx"

    if not config.exists() or not model_file.exists():
        fail("Triton repo structure is incorrect")

    ok(f"Triton repo: {repo}")
    ok(f"config.pbtxt: {config.stat().st_size} bytes")
    ok(f"model.onnx: {model_file.stat().st_size / 1024 / 1024:.1f} MB")


_triton_proc: subprocess.Popen | None = None


def start_and_test_triton() -> None:
    global _triton_proc

    step("7 / 7  Triton server + inference test")

    subprocess.run(["docker", "rm", "-f", TRITON_CONTAINER], capture_output=True)

    _triton_proc = subprocess.Popen(
        [
            "docker", "run", "--rm",
            "--name", TRITON_CONTAINER,
            "-p", f"{TRITON_HTTP_PORT}:8000",
            "-p", f"{TRITON_GRPC_PORT}:8001",
            "-p", f"{TRITON_METRICS_PORT}:8002",
            "-v", f"{ROOT}/triton_repo:/models",
            TRITON_IMAGE,
            "tritonserver", "--model-repository=/models",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"  Waiting for Triton at http://127.0.0.1:{TRITON_HTTP_PORT}...")
    ready = False
    for i in range(60):
        time.sleep(2)
        check = subprocess.run(
            ["curl", "-sf", f"http://127.0.0.1:{TRITON_HTTP_PORT}/v2/health/ready"],
            capture_output=True,
        )
        if check.returncode == 0:
            ready = True
            break
        if i % 5 == 0:
            print(f"  ... {i * 2}s elapsed")

    if not ready:
        fail("Triton did not become ready in 120 seconds")

    ok("Triton server ready")

    poetry("python", "scripts/test_triton.py")
    ok("Triton inference test passed")


def stop_triton() -> None:
    global _triton_proc
    subprocess.run(["docker", "stop", TRITON_CONTAINER], capture_output=True)
    if _triton_proc:
        _triton_proc.terminate()
        _triton_proc = None


def cleanup(*_) -> None:
    stop_triton()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        run_unit_tests()
        ensure_data()
        check_mlflow()
        train_one_epoch()
        export_and_infer_onnx()
        build_triton_repo()
        start_and_test_triton()

        print(f"\n{'=' * 60}")
        print("  ALL 7 STEPS COMPLETED SUCCESSFULLY")
        print(f"  Triton HTTP:  http://127.0.0.1:{TRITON_HTTP_PORT}")
        print(f"  MLflow runs:  {ROOT}/mlruns.db")
        print(f"  View UI:      mlflow server --backend-store-uri sqlite:///{ROOT}/mlruns.db")
        print(f"{'=' * 60}\n")

    except SystemExit:
        raise
    except Exception as e:
        print(f"\nFAILED: {e}")
        raise
    finally:
        stop_triton()
