from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from .gate_candidate_extractor import save_candidates
except Exception:
    try:
        from gate_candidate_extractor import save_candidates
    except Exception:
        save_candidates = None

IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)

def iou_xywh(a:Tuple[float,float,float,float], b:Tuple[float,float,float,float])->float:
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ax2, ay2 = ax+aw, ay+ah
    bx2, by2 = bx+bw, by+bh
    ix1, iy1 = max(ax,bx), max(ay,by)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    if inter<=0: return 0.0
    union = aw*ah + bw*bh - inter
    return inter/union if union>0 else 0.0

def nms(items:List[Dict[str,Any]], iou_thr:float=0.35)->List[Dict[str,Any]]:
    # items must have bbox_xywh and confidence
    items=sorted(items, key=lambda d: d["confidence"], reverse=True)
    kept=[]
    for it in items:
        bb=tuple(it["bbox_xywh"])
        if all(iou_xywh(bb, tuple(k["bbox_xywh"])) < iou_thr for k in kept):
            kept.append(it)
    return kept

def load_model(ckpt_path:Path, device:torch.device):
    ckpt=torch.load(str(ckpt_path), map_location="cpu")
    # Prefer checkpoint mapping (your trainer saves this)
    if "idx_to_label" not in ckpt:
        raise SystemExit("Checkpoint missing idx_to_label. Re-train with v2 trainer or provide mapping.")
    idx_to_label={int(k):v for k,v in ckpt["idx_to_label"].items()} if isinstance(next(iter(ckpt["idx_to_label"].keys())), str) else ckpt["idx_to_label"]
    img_size=int(ckpt.get("img_size",128))

    from torchvision import models
    import torch.nn as nn
    model=models.resnet18(weights=None)
    model.fc=nn.Linear(model.fc.in_features, len(idx_to_label))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()

    tfm=transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, idx_to_label, tfm

def predict_on_crops(model, idx_to_label, tfm, candidates:List[Dict[str,Any]], device, batch_size:int=64)->List[Dict[str,Any]]:
    out=[]
    # candidates expected to have crop_path and bbox_xywh (x,y,w,h)
    paths=[]
    bboxes=[]
    for c in candidates:
        if "crop_path" in c:
            p=Path(c["crop_path"])
            if p.exists():
                paths.append(p)
                bboxes.append(c.get("bbox_xywh") or c.get("bbox") or c.get("xywh"))
            else:
                paths.append(None)
                bboxes.append(None)
        else:
            paths.append(None); bboxes.append(None)

    idx_map=[i for i,p in enumerate(paths) if p is not None]
    valid_paths=[paths[i] for i in idx_map]
    valid_bboxes=[bboxes[i] for i in idx_map]

    for i in range(0,len(valid_paths),batch_size):
        batch_paths=valid_paths[i:i+batch_size]
        batch_boxes=valid_bboxes[i:i+batch_size]
        imgs=[]; keep=[]
        for p,bb in zip(batch_paths,batch_boxes):
            try:
                img=Image.open(p).convert("RGB")
                imgs.append(tfm(img))
                keep.append((p,bb))
            except Exception:
                continue
        if not imgs: 
            continue
        x=torch.stack(imgs,0).to(device)
        with torch.no_grad():
            probs=F.softmax(model(x), dim=1)
            conf, pred = probs.max(dim=1)
        for (p,bb),c,pr in zip(keep, conf.cpu().tolist(), pred.cpu().tolist()):
            out.append({
                "crop_path": str(p),
                "bbox_xywh": [float(v) for v in (bb if bb is not None else [0,0,0,0])],
                "pred_idx": int(pr),
                "pred_label": idx_to_label[int(pr)],
                "confidence": float(c),
            })
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image", help="QC image path")
    ap.add_argument("--candidates", help="Existing candidates.json")
    ap.add_argument("--model", required=True, help="gate_patch_model_v2.pt")
    ap.add_argument("--out", required=True, help="Output JSON")
    ap.add_argument("--tmp", default=".tmp_candidates", help="Temp folder")
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--min-conf", type=float, default=0.70)
    ap.add_argument("--iou", type=float, default=0.35, help="NMS IoU threshold")
    ap.add_argument("--keep-instances", action="store_true", help="Store instances in output")
    args=ap.parse_args()

    if not args.candidates and not args.image:
        raise SystemExit("Provide --image or --candidates")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_label, tfm = load_model(Path(args.model), device)

    # Load / create candidates
    if args.candidates:
        candidates_path=Path(args.candidates)
    else:
        if save_candidates is None:
            raise SystemExit("gate_candidate_extractor.py not importable; provide --candidates")
        tmp=Path(args.tmp); tmp.mkdir(parents=True, exist_ok=True)
        candidates_path=save_candidates(Path(args.image), tmp, pad=args.pad)

    meta=json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates=meta.get("candidates", [])

    preds=predict_on_crops(model, idx_to_label, tfm, candidates, device=device)

    # Per-class thresholds (tighten for confusing labels)
    strict = {"t":0.85, "s":0.85, "z":0.85, "p":0.85, "r":0.85}
    kept=[]
    for p in preds:
        lab=p["pred_label"]
        thr=strict.get(lab, args.min_conf)
        conf = float(p["confidence"])
        if conf >= thr and lab != "other":
            kept.append(p)

    # NMS de-dup per label
    dedup=[]
    by_label:Dict[str,List[Dict[str,Any]]]={}
    for k in kept:
        by_label.setdefault(k["pred_label"], []).append(k)
    for lab, lst in by_label.items():
        dedup.extend(nms(lst, iou_thr=args.iou))

    gates=sorted({d["pred_label"] for d in dedup})
    hist={}
    for d in dedup:
        hist[d["pred_label"]] = hist.get(d["pred_label"],0)+1

    out={
        "source_image": meta.get("source_image", args.image or ""),
        "candidates_json": str(candidates_path),
        "min_conf": args.min_conf,
        "iou_nms": args.iou,
        "quantum_gates": gates,
        "label_hist": dict(sorted(hist.items(), key=lambda kv:(-kv[1], kv[0]))),
        "counts": {
            "instances": len(dedup),
            "total_predictions": len(preds),
        }
    }
    if args.keep_instances:
        out["instances"]=dedup

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[DONE] wrote:", args.out)

if __name__=="__main__":
    main()
