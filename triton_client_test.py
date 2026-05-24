#!/usr/bin/env python3


import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tritonclient.http as httpclient
except ImportError:
    print("Installing tritonclient...")
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "tritonclient[all]", "-q"])
    import tritonclient.http as httpclient


def test_triton(model_name="waste_sort", url="localhost:8000"):

    with open("artifacts/class_map.json") as f:
        class_map = json.load(f)
    idx_to_class = {v: k for k, v in class_map.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    try:
        print(f"Connecting to Triton at {url}...")
        client = httpclient.InferenceServerClient(url=url)
        client.get_server_ready()
        print("Connected to Triton server")

    except Exception as e:
        print(f"Triton not ready: {e}")
        print("   Make sure Triton container is running:")
        print("   docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \\")
        print("     -v $PWD/triton_repo:/models \\")
        print("     nvcr.io/nvidia/tritonserver:24.07-py3 \\")
        print("     tritonserver --model-repository=/models")
        return False

    img_path = list(Path("data/test").glob("*.jpg"))[0]
    print(f"Testing on: {img_path.name}")

    img = Image.open(img_path).convert("RGB")
    scale = int(224 * 256 / 224)
    img = img.resize((scale, scale), Image.BILINEAR)
    left = (scale - 224) // 2
    top = (scale - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    print("Running Triton inference...")
    try:
        inputs = [httpclient.InferInput("input", arr.shape, "FP32")]
        inputs[0].set_data_from_numpy(arr)
        outputs = [httpclient.InferRequestedOutput("output")]

        result = client.infer(model_name, inputs=inputs, outputs=outputs)
        logits = result.as_numpy("output")

        probs = np.exp(logits[0] - np.max(logits[0]))
        probs = probs / probs.sum()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        print(f"\nTRITON INFERENCE SUCCESSFUL!")
        print(f"   Класс: {classes[pred_idx]}")
        print(f"   Уверенность: {confidence:.4f}")
        return True

    except Exception as e:
        print(f"Inference failed: {e}")
        return False


if __name__ == "__main__":
    success = test_triton()
    sys.exit(0 if success else 1)
