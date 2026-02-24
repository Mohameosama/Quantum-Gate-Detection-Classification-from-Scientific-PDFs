#!/usr/bin/env python3
# train_gate_patch_classifier_v2.py
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split_indices(labels: List[int], val_ratio: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for _, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(1, int(round(val_ratio * n))) if n >= 2 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_f1(cm: torch.Tensor) -> List[float]:
    f1s: List[float] = []
    for c in range(cm.size(0)):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(f1)
    return f1s


class GateCropDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


def build_transforms(img_size: int):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=4, translate=(0.05, 0.05), scale=(0.92, 1.08), shear=2),
        transforms.ColorJitter(brightness=0.18, contrast=0.18),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tfm, val_tfm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="labels.json")
    ap.add_argument("--out", default="gate_patch_model.pt", help="Output checkpoint")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--freeze-epochs", type=int, default=3, help="Train only classifier head for first N epochs")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--lr-head", type=float, default=3e-4, help="LR for head during frozen stage")
    ap.add_argument("--lr-ft", type=float, default=1e-4, help="LR for full fine-tune stage")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop", type=int, default=6, help="Stop if no val improvement for N epochs")
    args = ap.parse_args()

    set_seed(args.seed)

    labels_json = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    label_names = sorted({v["label"] for v in labels_json.values()})
    label_to_idx = {l: i for i, l in enumerate(label_names)}

    items: List[Tuple[str, int]] = []
    ys: List[int] = []
    for v in labels_json.values():
        p = Path(v["crop_path"])
        if p.exists():
            y = label_to_idx[v["label"]]
            items.append((str(p), y))
            ys.append(y)

    if len(items) < 300:
        raise SystemExit("Label more crops first (recommend >= 300).")

    train_tfm, val_tfm = build_transforms(args.img)
    base_ds_train = GateCropDataset(items, train_tfm)
    base_ds_val = GateCropDataset(items, val_tfm)

    train_idx, val_idx = stratified_split_indices(ys, val_ratio=args.val_ratio, seed=args.seed)
    train_ds = Subset(base_ds_train, train_idx)
    val_ds = Subset(base_ds_val, val_idx)

    train_labels = [ys[i] for i in train_idx]
    class_counts = torch.bincount(torch.tensor(train_labels), minlength=len(label_names)).float()
    class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
    sample_weights = class_weights[torch.tensor(train_labels)]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_idx), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(label_names))
    model.to(device)

    crit = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best = -1.0
    best_epoch = 0
    patience = 0

    # Stage 1: freeze backbone, train only head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.fc.parameters(), lr=args.lr_head)

    for ep in range(1, args.epochs + 1):
        if ep == args.freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr_ft)

        model.train()
        total = correct = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            total += x.size(0)

        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        model.eval()
        v_total = v_correct = 0
        y_true: List[int] = []
        y_pred: List[int] = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                v_correct += int((pred == y).sum().item())
                v_total += x.size(0)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())

        val_acc = v_correct / max(1, v_total)
        cm = confusion_matrix(y_true, y_pred, len(label_names))
        f1s = per_class_f1(cm)
        macro_f1 = sum(f1s) / len(f1s)

        stage = "head" if ep <= args.freeze_epochs else "finetune"
        print(
            f"epoch {ep:02d} [{stage}] train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} macro_f1={macro_f1:.3f}"
        )

        score = macro_f1 + 0.10 * val_acc
        if score > best + 1e-6:
            best = score
            best_epoch = ep
            patience = 0

            ckpt = {
                "model_state": model.state_dict(),
                "label_to_idx": label_to_idx,
                "idx_to_label": {i: l for l, i in label_to_idx.items()},
                "img_size": args.img,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "metrics": {"val_acc": val_acc, "macro_f1": macro_f1},
                "class_counts_train": {label_names[i]: int(class_counts[i].item()) for i in range(len(label_names))},
            }
            torch.save(ckpt, args.out)

            report_path = Path(args.out).with_suffix(".report.json")
            report = {
                "best_epoch": best_epoch,
                "val_acc": val_acc,
                "macro_f1": macro_f1,
                "labels": label_names,
                "per_class_f1": {label_names[i]: f1s[i] for i in range(len(label_names))},
                "confusion_matrix": cm.tolist(),
                "note": "cm[row=true, col=pred]",
            }
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        else:
            patience += 1

        if patience >= args.early_stop:
            print(f"[EARLY STOP] no improvement for {args.early_stop} epochs. best_epoch={best_epoch}")
            break

    print("[DONE] saved best:", args.out)
    print("best_epoch:", best_epoch)


if __name__ == "__main__":
    main()
