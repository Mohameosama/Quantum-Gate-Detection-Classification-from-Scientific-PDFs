from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix, classification_report


@dataclass
class EpochStats:
    loss: float
    acc: float


def get_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    # Train: stronger augmentation
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
        ], p=0.6),
        transforms.RandomAffine(
            degrees=4, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=2
        ),
        transforms.RandomPerspective(distortion_scale=0.10, p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Val/Test: no augmentation
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, eval_tf


def build_datasets(data_dir: str, img_size: int):
    train_tf, eval_tf = get_transforms(img_size)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_tf)

    # Expect class names: qc and non_qc (order may be alphabetical)
    # printing mapping so you know which index is which.
    return train_ds, val_ds, test_ds


def make_balanced_sampler(train_ds: datasets.ImageFolder) -> WeightedRandomSampler:
    # Compute class counts
    targets = torch.tensor([y for _, y in train_ds.samples], dtype=torch.long)
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / torch.clamp(class_counts.float(), min=1.0)

    # Weight each sample by its class weight
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )

    return model


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def run_one_epoch(model, loader, criterion, optimizer=None, device="cpu") -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return EpochStats(loss=total_loss / max(total, 1), acc=total_correct / max(total, 1))


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    all_y = []
    all_pred = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu()
        all_pred.extend(pred.tolist())
        all_y.extend(y.tolist())

    return all_y, all_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dataset folder containing train/val/test")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--freeze-backbone", action="store_true", help="freeze resnet backbone (recommended on CPU)")
    ap.add_argument("--unfreeze-after", type=int, default=6, help="epoch to unfreeze last layer block (optional)")
    ap.add_argument("--save-path", default="best_resnet18.pt")
    args = ap.parse_args()

    device = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    train_ds, val_ds, test_ds = build_datasets(args.data_dir, args.img_size)

    print("Class mapping:", train_ds.class_to_idx)
    num_classes = len(train_ds.classes)
    if num_classes != 2:
        raise SystemExit("Expected exactly 2 classes (qc and non_qc).")

    sampler = make_balanced_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False
    )

    model = build_model(num_classes=num_classes, freeze_backbone=args.freeze_backbone)
    model.to(device)

    # Class weights (still useful even with balanced sampler)
    train_targets = torch.tensor([y for _, y in train_ds.samples], dtype=torch.long)
    counts = torch.bincount(train_targets, minlength=num_classes).float()
    weights = (counts.sum() / torch.clamp(counts, min=1.0)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer only trains unfrozen params
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_epoch = -1

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Optional: unfreeze last block after some epochs
        if args.freeze_backbone and epoch == args.unfreeze_after:
            for name, p in model.named_parameters():
                # Unfreeze layer4 (last residual block) + fc
                if name.startswith("layer4") or name.startswith("fc"):
                    p.requires_grad = True
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 0.2)

        train_stats = run_one_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        val_stats = run_one_epoch(model, val_loader, criterion, optimizer=None, device=device)
        scheduler.step(val_stats.loss)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss={train_stats.loss:.4f} acc={train_stats.acc:.4f} | "
              f"val loss={val_stats.loss:.4f} acc={val_stats.acc:.4f}")

        if val_stats.loss < best_val_loss:
            best_val_loss = val_stats.loss
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "img_size": args.img_size
            }, args.save_path)

    print(f"\nBest val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Training time: {(time.time() - t0)/60:.1f} min")
    print(f"Saved best model to: {args.save_path}")

    # Test evaluation with best checkpoint
    ckpt = torch.load(args.save_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = evaluate(model, test_loader, device=device)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    labels = [idx_to_class[i] for i in range(num_classes)]

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))


if __name__ == "__main__":
    main()
