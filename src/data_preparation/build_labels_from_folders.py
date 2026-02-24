#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-root", required=True, help="Root folder with class subfolders")
    ap.add_argument("--out", required=True, help="Output labels.json path")
    args = ap.parse_args()

    root = Path(args.labeled_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    allowed = {
        "h","x","z","s","t","rx","ry","rz",
        "control_dot","target_plus","other"
    }

    labels = {}
    skipped = 0
    total = 0

    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        if label not in allowed:
            print(f"[WARN] Skipping unknown class folder: {label}")
            continue

        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                total += 1
                key = f"{label}__{p.stem}"
                labels[key] = {
                    "label": label,
                    "crop_path": str(p.resolve()),
                    "type": "patch",
                    "source_image": "",
                    "bbox": None,
                }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote {len(labels)} labels to {out} (total images scanned={total}, skipped={skipped})")

if __name__ == "__main__":
    main()

