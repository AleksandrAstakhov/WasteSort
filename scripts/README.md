# Utility Scripts

Simple Python scripts for common operations. No EOF, just plain commands.

## Available Scripts

### `check_system.py`
**Check system configuration and test all components.**

```bash
python3 scripts/check_system.py
```

Verifies:
- Dataset (classes, images count)
- Trained checkpoints (baseline, efficientnet)
- ONNX models
- Inference capability
- Triton server status
- MLflow integration

### `test_inference.py`
**Test inference on sample images.**

```bash
python3 scripts/test_inference.py
```

Tests:
- PyTorch checkpoint inference
- ONNX inference
- Accuracy on test images

### `export_onnx.py`
**Export EfficientNet model to ONNX format.**

```bash
python3 scripts/export_onnx.py
```

Exports:
- EfficientNet-B2 checkpoint to ONNX
- Saves to `artifacts/model.onnx` (1.46 MB)
- Shows compression ratio

### `test_triton.py`
**Test Triton Inference Server via HTTP API.**

```bash
python3 scripts/test_triton.py
```

Tests:
- Triton server connectivity
- Inference via HTTP API
- Accuracy on sample images

**Requires:** Triton server running
```bash
tritonserver --model-repository=$(pwd)/triton_repo
```

### `show_data_stats.py`
**Display dataset statistics and class mapping.**

```bash
python3 scripts/show_data_stats.py
```

Shows:
- Images per class
- Train/val/test split
- Class ID mapping
