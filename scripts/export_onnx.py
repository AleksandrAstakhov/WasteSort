#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from waste_sort.export.onnx_export import export_to_onnx


def find_latest_checkpoint(checkpoint_dir: str = "artifacts/checkpoints") -> str:

    checkpoints_dir = Path(checkpoint_dir)

    versioned = list(checkpoints_dir.glob("efficientnet-best-v*.ckpt"))

    unversioned = checkpoints_dir / "efficientnet-best.ckpt"

    if versioned:

        def get_version(path: Path) -> int:
            match = re.search(r"efficientnet-best-v(\d+)\.ckpt", path.name)
            return int(match.group(1)) if match else 0

        latest = max(versioned, key=get_version)
        return str(latest)

    if unversioned.exists():
        return str(unversioned)

    raise FileNotFoundError(
        f"No checkpoints found in '{checkpoints_dir}'. "
        f"Expected: efficientnet-best.ckpt or efficientnet-best-v*.ckpt"
    )


def main():
    print("Exporting EfficientNet to ONNX...\n")

    onnx_path = export_to_onnx(
        checkpoint_path=find_latest_checkpoint(),
        output_path="artifacts/model.onnx",
        image_size=224,
        opset_version=17,
    )

    print(f"ONNX model exported")

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"   Path: {onnx_path}")
    print(f"   Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
