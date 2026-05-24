#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-artifacts/model.onnx}"
TRT_PATH="${2:-artifacts/model.plan}"
TRT_IMAGE="nvcr.io/nvidia/tensorrt:24.07-py3"

echo "Converting ${ONNX_PATH} -> ${TRT_PATH}"
echo "Using Docker image: ${TRT_IMAGE}"

docker run --rm --gpus all \
    -v "$(pwd):/workspace" \
    -w /workspace \
    "${TRT_IMAGE}" \
    trtexec \
        --onnx="${ONNX_PATH}" \
        --saveEngine="${TRT_PATH}" \
        --fp16 \
        --optShapes=input:1x3x224x224 \
        --minShapes=input:1x3x224x224 \
        --maxShapes=input:8x3x224x224

echo "TensorRT engine saved to ${TRT_PATH}"
