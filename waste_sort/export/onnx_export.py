from __future__ import annotations

from pathlib import Path

import torch

from waste_sort.training.module import WasteClassifier


def export_to_onnx(
    checkpoint_path: str,
    output_path: str = "artifacts/model.onnx",
    image_size: int = 224,
    opset_version: int = 17,
) -> Path:
    """Export a Lightning checkpoint to ONNX.

    Args:
        checkpoint_path: Path to .ckpt file.
        output_path: Where to save the .onnx file.
        image_size: Input image size (H=W).
        opset_version: ONNX opset version.

    Returns:
        Path to the saved ONNX model.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    module = WasteClassifier.load_from_checkpoint(checkpoint_path)
    module.eval()
    model = module.model

    model = model.to("cpu")

    dummy_input = torch.randn(1, 3, image_size, image_size).to("cpu")

    torch.onnx.export(
        model,
        dummy_input,
        str(out),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"ONNX model exported to {out}")
    return out
