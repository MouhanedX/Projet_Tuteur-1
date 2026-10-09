"""Composite all 43 GTSRB classes onto captured CARLA sign faces.

The capture script writes raw_frames/ and capture_manifest.jsonl. This script
keeps the raw files untouched and writes full treated frames to treated_frames/.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_GTSRB")
RAW_DIR = os.path.join(BASE_DIR, "raw_frames")
TREATED_DIR = os.path.join(BASE_DIR, "treated_frames")
MANIFEST = os.path.join(BASE_DIR, "capture_manifest.jsonl")


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def load_test_pool():
    pool = defaultdict(list)
    with open(os.path.join(DATASET_DIR, "Test.csv"), encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cls = int(row["ClassId"])
            image = read_image(os.path.join(DATASET_DIR, *row["Path"].split("/")))
            if image is None:
                continue
            x1, y1 = int(row["Roi.X1"]), int(row["Roi.Y1"])
            x2, y2 = int(row["Roi.X2"]), int(row["Roi.Y2"])
            crop = image[y1:y2, x1:x2]
            if crop.size:
                pool[cls].append(cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA))
    missing = sorted(set(range(43)) - set(pool))
    if missing:
        raise RuntimeError(f"GTSRB test images are missing classes: {missing}")
    return pool


def brightness_match(image, target_mean):
    scale = np.clip(target_mean / max(float(image.mean()), 1e-3), 0.45, 3.0)
    adjusted = np.clip(image.astype(np.float32) * scale, 0, 255)
    source_mean = adjusted.mean() / 255.0
    target = target_mean / 255.0
    if source_mean < target * 0.9 and source_mean > 0.01:
        gamma = math.log(source_mean) / math.log(max(target, 0.01))
        adjusted = np.clip(255.0 * (adjusted / 255.0) ** gamma, 0, 255)
    return adjusted.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--output-dir", default=TREATED_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pool = load_test_pool()
    pointers = defaultdict(int)
    processed = 0

    with open(args.manifest, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            raw_path = os.path.join(args.raw_dir, record["file"])
            frame = read_image(raw_path)
            if frame is None:
                raise RuntimeError(f"Cannot read captured frame: {raw_path}")
            output = frame.copy()
            height, width = frame.shape[:2]
            for sign in record["signs"]:
                cls = int(sign["class"])
                points = np.float32(sign["quad"])
                art = pool[cls][pointers[cls] % len(pool[cls])]
                pointers[cls] += 1
                mask = np.zeros((height, width), np.uint8)
                cv2.fillPoly(mask, [points.astype(np.int32)], 255)
                region = frame[mask > 0]
                if region.size == 0:
                    continue
                matrix = cv2.getPerspectiveTransform(
                    np.float32([[0, 0], [128, 0], [128, 128], [0, 128]]), points
                )
                patch = cv2.warpPerspective(
                    brightness_match(art, float(region.mean())), matrix, (width, height)
                )
                feather = cv2.GaussianBlur(mask, (7, 7), 0)
                output[feather > 0] = patch[feather > 0]
            cv2.imwrite(os.path.join(args.output_dir, record["file"]), output)
            processed += 1
    print(f"Processed {processed} frames -> {args.output_dir}")


if __name__ == "__main__":
    main()
