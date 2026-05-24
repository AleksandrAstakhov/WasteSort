"""CLI entry point for WasteSort.

Usage:
    waste-sort train [overrides...]
    waste-sort infer
    waste-sort export_onnx
    waste-sort build_triton_repo
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from waste_sort.data.datamodule import WasteDataModule
from waste_sort.data.dataset import WasteDataset
from waste_sort.export.onnx_export import export_to_onnx
from waste_sort.inference.predictor import Predictor
from waste_sort.mlflow_utils import (
    end_run,
    init_mlflow_run,
    log_config,
    log_data_preparation,
    log_inference,
    log_model_registry,
    log_onnx_export,
    log_triton_repo_build,
)
from waste_sort.serving.triton import build_triton_repo
from waste_sort.training.module import WasteClassifier
from waste_sort.utils import plot_metrics


def _train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    run_id = init_mlflow_run(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        run_name=f"train-{cfg.model.name}",
        tags={"command": "train", "model": cfg.model.name},
    )

    try:
        log_config(OmegaConf.to_container(cfg, resolve=True))

        dm = WasteDataModule(
            data_dir=cfg.data.data_dir,
            image_size=cfg.data.image_size,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.data.num_workers,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            seed=cfg.seed,
            split_file=cfg.data.split_file,
        )

        WasteDataset.save_class_map(Path(cfg.data.class_map_file))

        log_data_preparation(
            split_file=cfg.data.split_file,
            class_map_file=cfg.data.class_map_file,
            data_dir=cfg.data.data_dir,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
        )

        model = WasteClassifier(
            model_name=cfg.model.name,
            num_classes=cfg.model.num_classes,
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            label_smoothing=cfg.optim.label_smoothing,
            scheduler_t_max=cfg.train.max_epochs,
            pretrained=cfg.model.pretrained,
            dropout=cfg.model.get("dropout", 0.3),
        )

        checkpoint_cb = ModelCheckpoint(
            dirpath=cfg.train.checkpoint_dir,
            filename=f"{cfg.model.name}-best",
            monitor="val/f1_macro",
            mode="max",
            save_top_k=1,
            verbose=True,
        )
        early_stop_cb = EarlyStopping(
            monitor="val/f1_macro",
            patience=cfg.train.patience,
            mode="max",
            verbose=True,
        )

        logger = MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.tracking_uri,
            run_name=f"train-{cfg.model.name}",
        )

        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            accelerator="auto",
            devices="auto",
            precision=cfg.train.precision,
            callbacks=[checkpoint_cb, early_stop_cb],
            logger=logger,
            deterministic=True,
            log_every_n_steps=10,
        )

        trainer.fit(model, datamodule=dm)

        trainer.test(model, datamodule=dm, ckpt_path="best")

        if Path(checkpoint_cb.best_model_path).exists():
            log_model_registry(
                checkpoint_path=checkpoint_cb.best_model_path,
                model_name=cfg.model.name,
            )

        plot_metrics(logger.run_id, cfg.mlflow.tracking_uri, cfg.model.name)

        print("\nTraining completed!")
        print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
        print(f"Best val/f1_macro: {checkpoint_cb.best_model_score:.4f}")
        print(f"MLflow run ID: {run_id}")

        end_run(status="success")

    except Exception as e:
        print(f"Training failed: {e}")
        end_run(status="failed")
        raise


def _infer(cfg: DictConfig) -> None:
    onnx_path = cfg.infer.get("onnx_path")
    ckpt_path = cfg.infer.get("checkpoint_path")

    init_mlflow_run(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        run_name="inference",
        tags={
            "command": "infer",
            "onnx_used": str(onnx_path is not None),
        },
    )

    try:
        log_config(OmegaConf.to_container(cfg, resolve=True))

        predictor = Predictor(
            onnx_path=onnx_path,
            checkpoint_path=ckpt_path,
            image_size=cfg.data.image_size,
        )
        predictor.predict_folder(
            folder=Path(cfg.infer.input_dir),
            output_csv=Path(cfg.infer.output_csv),
        )

        if Path(cfg.infer.output_csv).exists():
            df = pd.read_csv(cfg.infer.output_csv)
            mean_confidence = df["confidence"].mean() if "confidence" in df.columns else 0.0
            log_inference(
                predictions_csv=cfg.infer.output_csv,
                num_predictions=len(df),
                mean_confidence=mean_confidence,
                onnx_used=onnx_path is not None,
            )
            print(
                f"Inference completed: {len(df)} predictions, "
                f"mean confidence: {mean_confidence:.4f}"
            )

        end_run(status="success")

    except Exception as e:
        print(f"Inference failed: {e}")
        end_run(status="failed")
        raise


def _export_onnx(cfg: DictConfig) -> None:
    run_id = init_mlflow_run(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        run_name="export-onnx",
        tags={"command": "export_onnx"},
    )

    try:
        log_config(OmegaConf.to_container(cfg, resolve=True))

        print(f"Exporting ONNX model from: {cfg.export.checkpoint_path}")
        export_to_onnx(
            checkpoint_path=cfg.export.checkpoint_path,
            output_path=cfg.export.onnx_path,
            image_size=cfg.data.image_size,
        )

        log_onnx_export(
            onnx_path=cfg.export.onnx_path,
            checkpoint_path=cfg.export.checkpoint_path,
            input_size=(cfg.data.image_size, cfg.data.image_size),
        )

        print(f"ONNX export completed: {cfg.export.onnx_path}")
        print(f"MLflow run ID: {run_id}")

        end_run(status="success")

    except Exception as e:
        print(f"ONNX export failed: {e}")
        end_run(status="failed")
        raise


def _build_triton_repo(cfg: DictConfig) -> None:
    run_id = init_mlflow_run(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        run_name="build-triton",
        tags={"command": "build_triton_repo"},
    )

    try:
        log_config(OmegaConf.to_container(cfg, resolve=True))

        print(f"Building Triton model repository from: {cfg.export.onnx_path}")
        build_triton_repo(
            onnx_path=cfg.export.onnx_path,
            repo_path=cfg.serving.triton_repo,
            model_name=cfg.serving.model_name,
            max_batch_size=cfg.serving.max_batch_size,
            image_size=cfg.data.image_size,
        )

        log_triton_repo_build(
            triton_repo_path=cfg.serving.triton_repo,
            model_name=cfg.serving.model_name,
            onnx_path=cfg.export.onnx_path,
            max_batch_size=cfg.serving.max_batch_size,
        )

        print(f"Triton repository built: {cfg.serving.triton_repo}")
        print(f"MLflow run ID: {run_id}")
        print(
            f"\nTo start Triton server, run:\n"
            f"   docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 "
            f"-v $PWD/{cfg.serving.triton_repo}:/models "
            f"nvcr.io/nvidia/tritonserver:24.07-py3 "
            f"tritonserver --model-repository=/models"
        )

        end_run(status="success")

    except Exception as e:
        print(f"Triton build failed: {e}")
        end_run(status="failed")
        raise


COMMANDS = {
    "train": _train,
    "infer": _infer,
    "export_onnx": _export_onnx,
    "build_triton_repo": _build_triton_repo,
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    command = cfg.get("command")
    if command is None:
        for arg in sys.argv[1:]:
            if arg in COMMANDS:
                command = arg
                break

    if command not in COMMANDS:
        print("Usage: waste-sort <command> [overrides...]")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    print(f"Running command: {command}")
    print(OmegaConf.to_yaml(cfg))
    COMMANDS[command](cfg)


if __name__ == "__main__":
    main()
