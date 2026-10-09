"""Manually classify one image with the freshly trained 43-class CNN."""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from train_gtsrb import SignCNN, SIZE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE_DIR, "gtsrb_cnn_best.pt")
MANIFEST = os.path.join(BASE_DIR, "capture_manifest.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Classify one GTSRB/CARLA image")
    parser.add_argument("image", help="Path to a full frame or sign crop")
    parser.add_argument("--weights", default=WEIGHTS)
    args = parser.parse_args()
    if not os.path.isfile(args.image):
        raise FileNotFoundError(args.image)
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(
            f"Model weights not found: {args.weights}. Run train_gtsrb.py first."
        )
    image = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV cannot read image: {args.image}")
    model = SignCNN(n=43)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    crops = [(image, "provided image")]
    if os.path.isfile(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["file"] != os.path.basename(args.image):
                    continue
                crops = []
                for index, sign in enumerate(record["signs"]):
                    points = np.asarray(sign["quad"], dtype=np.float32)
                    x0, y0 = np.floor(points.min(axis=0)).astype(int)
                    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
                    crop = image[max(0, y0):min(image.shape[0], y1),
                                 max(0, x0):min(image.shape[1], x1)]
                    if crop.size:
                        crops.append((crop, f"sign {index}"))
                break
    if not crops:
        raise ValueError(f"No visible sign geometry found for {args.image}")
    tensors = [
        torch.from_numpy(cv2.resize(crop, (SIZE, SIZE))).float()
        .div(255).permute(2, 0, 1)
        for crop, _ in crops
    ]
    batch = torch.stack(tensors)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)
        confidences, predictions = probabilities.max(1)
    for (_, label), prediction, confidence in zip(crops, predictions, confidences):
        print(f"{label}: class={prediction.item()} confidence={confidence.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
