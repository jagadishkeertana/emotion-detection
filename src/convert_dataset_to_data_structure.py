# convert_dataset_to_data_structure.py
"""
Convert RAF-DB dataset or any emotion dataset into:
   data/train/<class>/
   data/val/<class>/

Usage:
  py convert_dataset_to_data_structure.py --src "C:\path\to\rafdb" --out "..\data" --split 0.85
"""

import argparse
import os
import shutil
import random
from pathlib import Path

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def ensure_dirs(out_dir: Path):
    """Create train and val directories with class folders."""
    for subset in ("train", "val"):
        for c in CLASSES:
            (out_dir / subset / c).mkdir(parents=True, exist_ok=True)


def find_class_dirs(src: Path):
    """Detect if dataset is structured by folders."""
    mappings = {}

    for c in CLASSES:
        folder1 = src / c
        folder2 = src / "train" / c

        if folder1.exists():
            mappings[c] = folder1
        elif folder2.exists():
            mappings[c] = folder2

    return mappings


def collect_from_class_folders(src_map, out_dir: Path, split_ratio):
    """Copy images into train/val folders."""
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"

    for cls, folder in src_map.items():
        files = [p for p in folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        random.shuffle(files)

        split = int(len(files) * split_ratio)
        train_files = files[:split]
        val_files = files[split:]

        for f in train_files:
            shutil.copy(f, train_dir / cls / f.name)

        for f in val_files:
            shutil.copy(f, val_dir / cls / f.name)

        print(f"[INFO] {cls}: train={len(train_files)}, val={len(val_files)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", type=float, default=0.85)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists():
        print("[ERROR] Source folder does not exist:", src)
        return

    ensure_dirs(out)
    src_map = find_class_dirs(src)

    if src_map:
        print("[INFO] Found_classes:", list(src_map.keys()))
        collect_from_class_folders(src_map, out, args.split)
    else:
        print("[ERROR] No class folders found in source directory. Expected:")
        print("Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral")


if __name__ == "__main__":
    main()
