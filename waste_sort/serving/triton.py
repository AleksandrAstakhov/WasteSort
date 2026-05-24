from __future__ import annotations

import shutil
from pathlib import Path

CONFIG_PBTXT_TEMPLATE = """\
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, {image_size}, {image_size} ]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ 12 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_AUTO
  }}
]
"""


def build_triton_repo(
    onnx_path: str,
    repo_path: str = "triton_repo",
    model_name: str = "waste_sort",
    max_batch_size: int = 8,
    image_size: int = 224,
) -> Path:
    """Create a Triton model repository from an ONNX model.

    Structure::

        triton_repo/
            waste_sort/
                config.pbtxt
                1/
                    model.onnx
    """
    repo = Path(repo_path)
    model_dir = repo / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(onnx_path, version_dir / "model.onnx")

    config_text = CONFIG_PBTXT_TEMPLATE.format(
        model_name=model_name,
        max_batch_size=max_batch_size,
        image_size=image_size,
    )
    (model_dir / "config.pbtxt").write_text(config_text)

    print(f"Triton model repository created at {repo}")
    print(f"  Model: {model_name}")
    print(f"  ONNX:  {version_dir / 'model.onnx'}")
    print(f"  Config: {model_dir / 'config.pbtxt'}")
    return repo
