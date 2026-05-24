#!/usr/bin/env python3


import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def setup_kaggle_token(token):
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    kaggle_config = kaggle_dir / "kaggle.json"

    if ":" not in token:
        print("Invalid token format. Expected: username:api_key")
        return False

    username, api_key = token.split(":", 1)

    config = {"username": username, "key": api_key}

    with open(kaggle_config, "w") as f:
        json.dump(config, f)

    os.chmod(kaggle_config, 0o600)
    print("Kaggle API token configured")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle garbage classification dataset")
    parser.add_argument("--token", type=str, help="Kaggle API token (username:api_key)")
    args = parser.parse_args()

    print("Downloading Kaggle dataset...\n")

    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_config.exists():
        if args.token:
            if not setup_kaggle_token(args.token):
                return False
        else:
            token = os.getenv("KAGGLE_API_TOKEN")
            if token:
                if not setup_kaggle_token(token):
                    return False
            else:
                print("Kaggle API not configured!")
                print("\nUsage options:")
                print(
                    "   1. Command line: python3 scripts/download_data.py --token username:api_key"
                )
                print("   2. Environment: export KAGGLE_API_TOKEN=username:api_key")
                print("   3. Manual setup:")
                print("      - Get token from: https://www.kaggle.com/settings/account")
                print("      - Save to: ~/.kaggle/kaggle.json")
                print("      - Run: chmod 600 ~/.kaggle/kaggle.json")
                return False

    print("Downloading from Kaggle (this may take a few minutes)...")
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "mostafaabla/garbage-classification",
            "--unzip",
            "-p",
            "data/",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Download failed: {result.stderr}")
        return False

    print("Download complete")

    print("\nOrganizing files...")
    src = Path("data/garbage_classification")
    dst = Path("data/raw")

    if not src.exists():
        print(f"Source directory not found: {src}")
        return False

    if dst.exists():
        shutil.rmtree(dst)

    src.rename(dst)
    print("Downloaded 12 classes")

    print("\n Consolidating 12 10 classes...")

    for glass_variant in ["brown-glass", "green-glass", "white-glass"]:
        src_glass = dst / glass_variant
        dst_glass = dst / "glass"

        if src_glass.exists():
            dst_glass.mkdir(exist_ok=True)
            for img in src_glass.glob("*.jpg"):
                shutil.copy2(img, dst_glass / f"{img.name}")
            shutil.rmtree(src_glass)
            print(f"   Merged {glass_variant} glass")

    src_battery = dst / "battery"
    if src_battery.exists():
        dst_unknown = dst / "unknown"
        dst_unknown.mkdir(exist_ok=True)
        for img in src_battery.glob("*.jpg"):
            shutil.copy2(img, dst_unknown / img.name)
        shutil.rmtree(src_battery)
        print("Mapped battery unknown")

    classes = len(list(dst.iterdir()))
    images = sum(len(list((dst / d).glob("*.jpg"))) for d in dst.iterdir() if d.is_dir())

    shutil.move(str(dst / "metal"), str(dst / "metals"))

    print("\nDataset ready!")
    print(f"   Classes: {classes} (consolidated from 12)")
    print(f"   Images: {images}")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
