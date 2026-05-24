#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path


def main():
    print("Dataset Statistics\n" + "=" * 50)

    print("\nClasses in data/raw/:")
    data_root = Path("data/raw")

    class_counts = defaultdict(int)
    total = 0
    for class_dir in sorted(data_root.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            class_counts[class_dir.name] = count
            total += count
            print(f"  {class_dir.name:15} : {count:5} images")

    print(f"\n  {'TOTAL':15} : {total:5} images")

    print("\nTrain/Val/Test Split:")
    split_file = Path("artifacts/split.json")
    if split_file.exists():
        with open(split_file) as f:
            split = json.load(f)
            train_count = len(split["train"])
            val_count = len(split["val"])
            test_count = len(split["test"])
            total_split = train_count + val_count + test_count

            print(f"  Train: {train_count:5} ({100*train_count/total_split:.1f}%)")
            print(f"  Val:   {val_count:5} ({100*val_count/total_split:.1f}%)")
            print(f"  Test:  {test_count:5} ({100*test_count/total_split:.1f}%)")
            print(f"  Total: {total_split:5}")

    print("\nClass ID Mapping:")
    class_map_file = Path("artifacts/class_map.json")
    if class_map_file.exists():
        with open(class_map_file) as f:
            class_map = json.load(f)
            for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
                print(f"  {idx}: {name}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
