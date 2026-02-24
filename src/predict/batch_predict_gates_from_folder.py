#!/usr/bin/env python3

from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

IMG_EXTS={".png",".jpg",".jpeg",".webp",".tif",".tiff"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-conf", type=float, default=0.70)
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--keep-instances", action="store_true")
    args=ap.parse_args()

    images_dir=Path(args.images_dir)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    imgs=[p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort()
    if args.limit>0:
        imgs=imgs[:args.limit]

    merged={"images_dir": str(images_dir), "model": str(Path(args.model)), "min_conf": args.min_conf, "results": {}}

    script = str(Path(__file__).with_name("predict_gates_from_qc_image.py"))

    for img in imgs:
        out_json = out_dir/(img.stem + ".gates.json")
        cmd=["python", script,
             "--image", str(img),
             "--model", str(args.model),
             "--out", str(out_json),
             "--min-conf", str(args.min_conf),
             "--iou", str(args.iou)]
        if args.keep_instances:
            cmd.append("--keep-instances")
        try:
            subprocess.run(cmd, check=True)
            merged["results"][img.name]=str(out_json)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {img.name} -> {e}")
            if not args.continue_on_error:
                raise

    merged_path = out_dir/"gates_merged_index.json"
    merged_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("[DONE] merged index:", merged_path)

if __name__=="__main__":
    main()
