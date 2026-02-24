"""
gate_candidate_extractor.py

Extract candidate gate patches (rectangular gate boxes + control/target symbols)
from a quantum-circuit figure image.

Outputs:
- crops/ : PNG crops
- candidates.json : metadata for each crop (bbox, type, source image, etc.)

This is the first step in the "hybrid vision + small model" approach:
1) detect candidate regions with classic CV
2) label them (manual)
3) train a small classifier on the crops
4) run classifier on new PDFs and infer gates
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Literal

import cv2
import numpy as np


def _to_py(o):
    """Convert numpy scalars/arrays inside nested structures into JSON-safe Python types."""
    import numpy as _np
    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(x) for x in o]
    return o


CandidateType = Literal["gate_box", "control_dot", "target_plus", "meter", "unknown_symbol"]

@dataclass
class Candidate:
    id: str
    type: CandidateType
    bbox: Tuple[int, int, int, int]  # x,y,w,h in original image
    area: int
    score: float
    crop_path: str

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _preprocess(gray: np.ndarray) -> np.ndarray:
    # Normalize contrast
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    # Binary inverted (foreground = white)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    return th

def detect_gate_boxes(img_bgr: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
    """Detect rectangular gate boxes."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    th = _preprocess(gray)

    # Close gaps in rectangle edges
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    boxes = []

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < 300 or area > 0.30*W*H:
            continue
        ar = w / float(h + 1e-6)
        # gate boxes are usually close to square/rect
        if ar < 0.5 or ar > 3.2:
            continue
        if w < 18 or h < 18:
            continue
        # Reject very elongated wire segments
        if w > 0.60*W or h > 0.60*H:
            continue

        # Score: prefer medium rectangles
        score = 1.0
        if 0.8 <= ar <= 1.8:
            score += 0.5
        if 700 <= area <= 20000:
            score += 0.5
        boxes.append((x,y,w,h,score))

    # Sort by score then area
    boxes.sort(key=lambda b: (b[4], b[2]*b[3]), reverse=True)

    # Deduplicate by IoU
    def iou(a,b):
        ax,ay,aw,ah,_ = a
        bx,by,bw,bh,_ = b
        ax1, ay1 = ax+aw, ay+ah
        bx1, by1 = bx+bw, by+bh
        inter_w = max(0, min(ax1,bx1)-max(ax,bx))
        inter_h = max(0, min(ay1,by1)-max(ay,by))
        inter = inter_w*inter_h
        union = aw*ah + bw*bh - inter
        return 0 if union<=0 else inter/union

    kept = []
    for b in boxes:
        if all(iou(b,k) < 0.55 for k in kept):
            kept.append(b)
        if len(kept) >= 200:
            break
    return kept

def detect_control_dots(img_bgr: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
    """Detect small filled circles (control dots)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    th = _preprocess(gray)

    # Remove long wires to isolate blobs
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    th2 = cv2.subtract(th, cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1))
    th2 = cv2.subtract(th2, cv2.morphologyEx(th2, cv2.MORPH_OPEN, v_kernel, iterations=1))

    cnts, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    dots = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < 30 or area > 2000:
            continue
        ar = w/float(h+1e-6)
        if ar < 0.6 or ar > 1.6:
            continue
        # prefer very small
        score = 1.0
        if area < 400:
            score += 0.7
        dots.append((x,y,w,h,score))
    dots.sort(key=lambda b: (b[4], -(b[2]*b[3])), reverse=True)
    return dots[:400]

def detect_target_plus(img_bgr: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
    """
    Detect target '⊕' symbols approximately:
    - find circles
    - within circle, look for cross-like strokes
    This is approximate and will be refined later.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # use edges for Hough
    edges = cv2.Canny(gray, 80, 200)

    H, W = gray.shape[:2]
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=18,
                               param1=120, param2=16, minRadius=6, maxRadius=26)
    out = []
    if circles is None:
        return out
    circles = np.around(circles[0, :]).astype(int)
    for (cx, cy, r) in circles:
        # ensure Python ints (avoid uint16 overflow)
        cx, cy, r = int(cx), int(cy), int(r)
        x0 = max(0, cx - r - 6)
        y0 = max(0, cy - r - 6)
        x1 = min(W, cx + r + 6)
        y1 = min(H, cy + r + 6)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = gray[y0:y1, x0:x1]
        if crop is None or crop.size == 0:
            continue

        # quick check for plus: count vertical/horizontal strokes
        th = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        if th is None or th.size == 0:
            continue
        # projections
        v = th.sum(axis=0)
        h = th.sum(axis=1)
        v_peak = (v.max() > 3.5 * (v.mean()+1e-6))
        h_peak = (h.max() > 3.5 * (h.mean()+1e-6))
        if v_peak and h_peak:
            w = x1-x0
            h2 = y1-y0
            score = 1.0 + 0.7
            out.append((x0,y0,w,h2,score))
    out.sort(key=lambda b: b[4], reverse=True)
    return out[:250]

def save_candidates(img_path: Path, out_dir: Path, pad: int = 6) -> Path:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise SystemExit(f"Failed to read image: {img_path}")

    crops_dir = out_dir / "crops"
    _ensure_dir(crops_dir)

    candidates: List[Candidate] = []
    H, W = img_bgr.shape[:2]

    # Gate boxes
    for i,(x,y,w,h,score) in enumerate(detect_gate_boxes(img_bgr)):
        x0 = max(0, x-pad); y0 = max(0, y-pad)
        x1 = min(W, x+w+pad); y1 = min(H, y+h+pad)
        crop = img_bgr[y0:y1, x0:x1]
        cid = f"gate_{i:04d}"
        crop_path = crops_dir / f"{cid}.png"
        cv2.imwrite(str(crop_path), crop)
        candidates.append(Candidate(id=cid, type="gate_box", bbox=(x0,y0,x1-x0,y1-y0),
                                    area=(x1-x0)*(y1-y0), score=float(score), crop_path=str(crop_path)))

    # Control dots
    for i,(x,y,w,h,score) in enumerate(detect_control_dots(img_bgr)):
        x0 = max(0, x-pad); y0 = max(0, y-pad)
        x1 = min(W, x+w+pad); y1 = min(H, y+h+pad)
        crop = img_bgr[y0:y1, x0:x1]
        cid = f"dot_{i:04d}"
        crop_path = crops_dir / f"{cid}.png"
        cv2.imwrite(str(crop_path), crop)
        candidates.append(Candidate(id=cid, type="control_dot", bbox=(x0,y0,x1-x0,y1-y0),
                                    area=(x1-x0)*(y1-y0), score=float(score), crop_path=str(crop_path)))

    # Target plus
    for i,(x,y,w,h,score) in enumerate(detect_target_plus(img_bgr)):
        x0 = max(0, x-pad); y0 = max(0, y-pad)
        x1 = min(W, x+w+pad); y1 = min(H, y+h+pad)
        crop = img_bgr[y0:y1, x0:x1]
        cid = f"plus_{i:04d}"
        crop_path = crops_dir / f"{cid}.png"
        cv2.imwrite(str(crop_path), crop)
        candidates.append(Candidate(id=cid, type="target_plus", bbox=(x0,y0,x1-x0,y1-y0),
                                    area=(x1-x0)*(y1-y0), score=float(score), crop_path=str(crop_path)))

    # Write metadata
    meta = {
        "source_image": str(img_path),
        "out_dir": str(out_dir),
        "num_candidates": len(candidates),
        "candidates": [asdict(c) for c in candidates],
        "notes": {
            "bbox": "x,y,w,h in original image pixels",
            "types": ["gate_box","control_dot","target_plus"],
        }
    }
    out_json = out_dir / "candidates.json"
    meta = _to_py(meta)
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a QC group image (PNG).")
    ap.add_argument("--out", required=True, help="Output directory for crops + json.")
    ap.add_argument("--pad", type=int, default=6, help="Padding around bboxes.")
    args = ap.parse_args()

    img_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = save_candidates(img_path, out_dir, pad=args.pad)
    print("[DONE] Wrote:", out_json)

if __name__ == "__main__":
    main()
