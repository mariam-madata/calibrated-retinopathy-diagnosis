"""
Extract raw logits from your ACTUAL trained ResNet18 diabetic retinopathy
model, to replace the synthetic demo logits with real numbers.

Run this where you have the trained checkpoint and the APTOS dataset
available (e.g. Google Colab, or wherever `diabetic-retinopathy-detection`
was originally trained) -- NOT in an environment without GPU/dataset access.

Usage:
    python extract_logits_from_model.py \
        --checkpoint path/to/best_model.pth \
        --data-dir path/to/aptos_val_split \
        --split val \
        --out-dir cached_logits/

Repeat with --split test and the test directory to get test logits too.
The saved val_logits/val_labels are used to FIT the temperature; the saved
test_logits/test_labels are used to EVALUATE calibration on unseen data.
Never fit temperature on the test split -- that would leak information.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def build_model(num_classes: int = 5) -> nn.Module:
    """Recreates the exact architecture used in diabetic-retinopathy-detection:
    ResNet18 backbone with a replaced final linear layer.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_eval_transform() -> transforms.Compose:
    # Must match the eval-time transform used during original training
    # (see diabetic-retinopathy-detection/src/dataset.py::get_transforms).
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def extract_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pth state_dict")
    parser.add_argument("--data-dir", required=True, help="ImageFolder-style directory for this split")
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--out-dir", default="cached_logits")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=args.num_classes).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    dataset = datasets.ImageFolder(args.data_dir, transform=get_eval_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    logits, labels = extract_logits(model, loader, device)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, f"{args.split}_logits.npy"), logits)
    np.save(os.path.join(args.out_dir, f"{args.split}_labels.npy"), labels)

    print(f"Saved {len(labels)} real {args.split} logits to {args.out_dir}/")
    print(f"Class order (ImageFolder): {dataset.classes}")


if __name__ == "__main__":
    main()
