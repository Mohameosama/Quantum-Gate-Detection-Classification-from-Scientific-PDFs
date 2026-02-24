#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from predict.gate_candidate_extractor import save_candidates



IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def iter_images(root: Path) -> List[Path]:
    imgs = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    imgs.sort()
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-root", required=True, help="Folder with QC images")
    ap.add_argument("--out-root", required=True, help="Output root for all crops and merged index")
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    qc_root = Path(args.qc_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    images = iter_images(qc_root)
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    print(f"[INFO] Found {len(images)} images under {qc_root}")

    merged: Dict[str, Any] = {
        "qc_root": str(qc_root),
        "out_root": str(out_root),
        "images": [],
    }

    for idx, img_path in enumerate(images, 1):
        rel = img_path.relative_to(qc_root)
        stem = str(rel).replace("/", "__").replace("\\", "__")
        per_img_out = out_root / "per_image" / stem
        per_img_out.mkdir(parents=True, exist_ok=True)

        try:
            cand_json = save_candidates(img_path, per_img_out, pad=args.pad)
            meta = json.loads(Path(cand_json).read_text(encoding="utf-8"))
            merged["images"].append(
                {
                    "image": str(img_path),
                    "relative": str(rel),
                    "candidates_json": str(cand_json),
                    "num_candidates": meta.get("num_candidates", 0),
                }
            )
            if idx % 25 == 0:
                print(f"[INFO] {idx}/{len(images)} processed...")
        except Exception as e:
            print(f"[WARN] Failed on {img_path}: {e}")

    merged_path = out_root / "candidates_all.json"
    merged_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("[DONE] Merged index:", merged_path)


if __name__ == "__main__":
    main()
