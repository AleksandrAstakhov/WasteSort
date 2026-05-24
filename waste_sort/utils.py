from __future__ import annotations

from pathlib import Path


def plot_metrics(
    run_id: str,
    tracking_uri: str,
    model_name: str,
    output_dir: str = "plots",
) -> None:
    try:
        import matplotlib.pyplot as plt
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        metrics_to_plot = {
            "train/loss_epoch": "Training Loss",
            "val/loss": "Validation Loss",
            "val/f1_macro": "Validation Macro F1",
            "val/accuracy": "Validation Accuracy",
        }

        for metric_key, title in metrics_to_plot.items():
            try:
                history = client.get_metric_history(run_id, metric_key)
                if not history:
                    continue
                steps = [m.step for m in history]
                values = [m.value for m in history]

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(steps, values)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric_key.split("/")[-1])
                ax.set_title(f"{title} ({model_name})")
                ax.grid(True, alpha=0.3)

                fname = f"{model_name}_{metric_key.replace('/', '_')}.png"
                fig.savefig(out / fname, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved plot: {out / fname}")
            except Exception:
                continue

    except ImportError:
        print("matplotlib or mlflow not installed, skipping plots.")
