from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow


def init_mlflow_run(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    tags: dict[str, str] | None = None,
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        run_id = run.info.run_id
        print(f"MLflow run started: {run_name} (ID: {run_id})")
        return run_id


def log_data_preparation(
    split_file: str,
    class_map_file: str,
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> None:
    mlflow.log_param("data_preparation/data_dir", data_dir)
    mlflow.log_param("data_preparation/train_ratio", train_ratio)
    mlflow.log_param("data_preparation/val_ratio", val_ratio)
    mlflow.log_param("data_preparation/test_ratio", 1.0 - train_ratio - val_ratio)

    if Path(split_file).exists():
        with open(split_file) as f:
            split = json.load(f)
        mlflow.log_param("data_preparation/train_samples", len(split["train"]))
        mlflow.log_param("data_preparation/val_samples", len(split["val"]))
        mlflow.log_param("data_preparation/test_samples", len(split["test"]))

        mlflow.log_artifact(split_file, artifact_path="data")

    if Path(class_map_file).exists():
        mlflow.log_artifact(class_map_file, artifact_path="data")
        with open(class_map_file) as f:
            class_map = json.load(f)
        mlflow.log_param("data_preparation/num_classes", len(class_map))


def log_inference(
    predictions_csv: str,
    num_predictions: int,
    mean_confidence: float,
    onnx_used: bool = False,
) -> None:
    mlflow.log_param("inference/num_predictions", num_predictions)
    mlflow.log_metric("inference/mean_confidence", mean_confidence)
    mlflow.log_param("inference/onnx_used", onnx_used)

    if Path(predictions_csv).exists():
        mlflow.log_artifact(predictions_csv, artifact_path="inference")
        print(f"Logged inference results: {predictions_csv}")


def log_onnx_export(
    onnx_path: str,
    checkpoint_path: str,
    input_size: tuple[int, int] = (224, 224),
) -> None:
    if not Path(onnx_path).exists():
        print(f"ONNX file not found: {onnx_path}")
        return

    size_mb = Path(onnx_path).stat().st_size / (1024**2)
    mlflow.log_param("export/onnx_input_height", input_size[0])
    mlflow.log_param("export/onnx_input_width", input_size[1])
    mlflow.log_metric("export/onnx_size_mb", size_mb)
    mlflow.log_param("export/source_checkpoint", Path(checkpoint_path).name)

    mlflow.log_artifact(onnx_path, artifact_path="models")
    print(f"Logged ONNX export ({size_mb:.2f} MB)")


def log_triton_repo_build(
    triton_repo_path: str,
    model_name: str,
    onnx_path: str,
    max_batch_size: int = 8,
) -> None:
    mlflow.log_param("serving/triton_repo", triton_repo_path)
    mlflow.log_param("serving/model_name", model_name)
    mlflow.log_param("serving/max_batch_size", max_batch_size)

    config_path = Path(triton_repo_path) / model_name / "config.pbtxt"
    if config_path.exists():
        mlflow.log_artifact(str(config_path), artifact_path="serving")

    print(f"Logged Triton model repository: {triton_repo_path}")


def log_model_registry(
    checkpoint_path: str,
    model_name: str,
    framework: str = "pytorch-lightning",
) -> None:
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    size_mb = Path(checkpoint_path).stat().st_size / (1024**2)
    mlflow.log_param("model/name", model_name)
    mlflow.log_param("model/framework", framework)
    mlflow.log_metric("model/checkpoint_size_mb", size_mb)
    mlflow.log_artifact(checkpoint_path, artifact_path="models")


def log_config(config_dict: dict[str, Any]) -> None:
    for key, value in _flatten_dict(config_dict).items():
        if isinstance(value, (str, int, float, bool)):
            mlflow.log_param(key, value)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def end_run(status: str = "success") -> None:
    mlflow.set_tag("run_status", status)
    mlflow.end_run()
    print(f"MLflow run ended with status: {status}")
