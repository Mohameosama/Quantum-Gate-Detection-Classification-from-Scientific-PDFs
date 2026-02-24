"""
predict_folder.py

Run inference on a folder of images using a trained ResNet model.

Compatible with checkpoints saved as:
{
    "model_state": state_dict,
    "class_to_idx": {...},
    "img_size": int
}

Usage:
python predict_folder.py --dir output --ckpt best_resnet18.pt --threshold 0.8
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ----------------------------
# Model definition (MUST match training)
# ----------------------------

def build_model(num_classes: int = 2):
    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model


# ----------------------------
# Image preprocessing
# ----------------------------

def get_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ----------------------------
# Prediction
# ----------------------------

@torch.no_grad()
def predict_image(model, image_path, transform, device):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0, 1].item()

    return prob


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Folder with images")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cpu")

    # ---- Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    img_size = ckpt.get("img_size", 224)

    # ---- Build & load model
    model = build_model(num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    transform = get_transform(img_size)

    # ---- Run prediction
    image_paths = sorted([
        p for p in Path(args.dir).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])

    if not image_paths:
        print("No images found.")
        return

    print(f"\nPredicting on {len(image_paths)} images\n")

    for path in image_paths:
        prob_qc = predict_image(model, path, transform, device)
        label = "qc" if prob_qc >= args.threshold else "non_qc"

        print(
            f"{path.name:40s} "
            f"prob_qc={prob_qc:.3f} "
            f"→ {label}"
        )


if __name__ == "__main__":
    main()
