# step_enrich_from_layout.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image

# your extraction class/function
try:
    from pipeline.extract_figures import extract_figures_from_pdf, ExtractedFigure
except Exception:
    from ..pipeline.extract_figures import extract_figures_from_pdf, ExtractedFigure

# ----------------------------
# Canonical paper text + spans
# ----------------------------

def build_paper_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    chunks = []
    for i in range(len(doc)):
        page_no = i + 1
        chunks.append(f"\n\n===== PAGE {page_no} =====\n\n")
        chunks.append(doc[i].get_text("text") or "")
    doc.close()
    return "".join(chunks)

def find_spans(paper_text: str, snippets: List[str]) -> List[Tuple[int, int]]:
    spans = []
    for s in snippets:
        if not s:
            spans.append((-1, -1))
            continue
        i = paper_text.find(s)
        spans.append((i, i + len(s)) if i != -1 else (-1, -1))
    return spans

# ----------------------------
# Layout-aware caption linking
# ----------------------------

FIG_RE = re.compile(r"(?i)\b(fig\.?|figure)\s*(\d+)\b")

def _x_overlap(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return 0.0 if union <= 0 else inter / union

def get_text_blocks(page: fitz.Page) -> List[Tuple[float, float, float, float, str]]:
    # blocks: (x0, y0, x1, y1, "text", block_no, block_type)
    blocks = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        t = (txt or "").strip()
        if t:
            blocks.append((x0, y0, x1, y1, t))
    return blocks

def link_caption_by_bbox(page: fitz.Page, bbox: Tuple[float,float,float,float]) -> Tuple[Optional[int], Optional[str]]:
    """
    Find caption text linked to image bbox:
    - Prefer blocks BELOW the image (y0 >= image_y1 - small slack)
    - Prefer strong x-overlap with image bbox
    - Prefer blocks containing "Fig/Figure N"
    """
    ix0, iy0, ix1, iy1 = bbox
    blocks = get_text_blocks(page)

    # candidate blocks below image
    candidates = []
    for x0, y0, x1, y1, txt in blocks:
        # below (allow small overlap because some layouts touch)
        if y0 >= (iy1 - 3.0):
            xo = _x_overlap(ix0, ix1, x0, x1)
            if xo >= 0.25:  # require some horizontal alignment
                dist = y0 - iy1
                has_fig = bool(FIG_RE.search(txt))
                # scoring: prefer fig blocks, close distance, good x-overlap
                score = (3.0 if has_fig else 0.0) + (2.0 * xo) - (0.01 * dist)
                candidates.append((score, dist, xo, txt))

    if not candidates:
        # fallback: any block with Fig N on page
        fig_blocks = [t for *_, t in blocks if FIG_RE.search(t)]
        if fig_blocks:
            txt = fig_blocks[0]
            m = FIG_RE.search(txt)
            return (int(m.group(2)) if m else None), txt
        return None, None

    candidates.sort(reverse=True)
    caption = candidates[0][3]
    m = FIG_RE.search(caption)
    fig_no = int(m.group(2)) if m else None
    return fig_no, caption

def extract_context_paragraphs(page_text: str, fig_no: int, max_paras: int = 2) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
    pat = re.compile(rf"(?i)\b(fig\.?|figure)\s*{fig_no}\b")
    ctx = [p for p in paras if pat.search(p)]
    return ctx[:max_paras]

# ----------------------------
# Gates & problem (baseline + optional OCR)
# ----------------------------

GATE_ALIASES = {
    "h": "h", "hadamard": "h",
    "x": "x", "y": "y", "z": "z",
    "s": "s", "t": "t",
    "rx": "rx", "ry": "ry", "rz": "rz",
    "u1": "u1", "u2": "u2", "u3": "u3",
    "cx": "cx", "cnot": "cx",
    "cz": "cz",
    "ccx": "ccx", "toffoli": "ccx",
    "ccnot": "ccx",  # normalize to CCX
    "swap": "swap", "iswap": "iswap",
    "measure": "measure", "measurement": "measure",
}
TOK_RE = re.compile(r"(?i)\b[a-z][a-z0-9]{0,10}\b")

FIN_PROBLEM_PAT = re.compile(r"(?i)\b(payoff|strike price|option pricing|pricing|financial)\b")

def gates_from_text(descriptions: List[str]) -> List[str]:
    joined = "\n".join(descriptions).lower()
    found = set()
    for tok in TOK_RE.findall(joined):
        if tok in GATE_ALIASES:
            found.add(GATE_ALIASES[tok])
    return sorted(g for g in found if g)

def problem_from_text(descriptions: List[str]) -> str:
    joined = "\n".join(descriptions)
    if FIN_PROBLEM_PAT.search(joined):
        return "Quantum payoff computation / option pricing (financial analysis)"
    # (keep your other algorithm patterns here if you want)
    return ""

def gates_from_ocr(image_path: str) -> List[str]:
    """
    Optional OCR: if pytesseract is installed, attempt to read gate labels from the diagram.
    If not installed, returns empty list.
    """
    try:
        import pytesseract
    except Exception:
        return []

    img = Image.open(image_path).convert("RGB")

    # Simple OCR pass (we’ll refine boxes later):
    text = pytesseract.image_to_string(img, config="--psm 6")
    text = text.upper()

    # capture common gate tokens (extend as you like)
    candidates = set()
    for token in re.findall(r"\b[A-Z]{1,6}\b", text):
        t = token.lower()
        if t in GATE_ALIASES:
            candidates.add(GATE_ALIASES[t])
        # common OCR misspells
        if t in {"conot", "ccnot"}:
            candidates.add("ccx")

    return sorted(c for c in candidates if c)

# ----------------------------
# Main: build dataset JSON for a PDF
# ----------------------------

def build_dataset_for_pdf(pdf_path: str, out_json: str, images_dir: str) -> None:
    pdf_path = str(pdf_path)
    out_json = str(out_json)
    images_dir = str(images_dir)

    # Extract figures with bbox/xref (your updated extractor)
    figures = extract_figures_from_pdf(pdf_path=pdf_path, output_dir=images_dir, min_size=50)

    paper_text = build_paper_text(pdf_path)
    doc = fitz.open(pdf_path)

    dataset: Dict[str, Dict[str, Any]] = {}

    for fig in figures:
        if not fig.bbox:
            continue

        page = doc[fig.page_number - 1]
        page_text = page.get_text("text") or ""

        fig_no, caption = link_caption_by_bbox(page, fig.bbox)

        descriptions: List[str] = []
        if caption:
            descriptions.append(caption)

        if fig_no is not None:
            descriptions.extend(extract_context_paragraphs(page_text, fig_no, max_paras=2))

        # de-dup descriptions
        descriptions = [d for i, d in enumerate(descriptions) if d and d not in descriptions[:i]]

        spans = find_spans(paper_text, descriptions)

        # gates: OCR first, fallback to text
        gates = gates_from_ocr(fig.file_path)
        if not gates:
            gates = gates_from_text(descriptions)

        problem = problem_from_text(descriptions)

        key = Path(fig.file_path).name
        dataset[key] = {
            "arxiv_number": fig.pdf_name,
            "page_number": fig.page_number,
            "figure_number": fig_no,  # may be None
            "quantum_gates": gates,
            "quantum_problem": problem,
            "descriptions": descriptions,
            "text_positions": spans,
        }

    doc.close()

    Path(out_json).write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--images-dir", required=True)
    args = ap.parse_args()

    build_dataset_for_pdf(args.pdf, args.out_json, args.images_dir)
    print(f"[DONE] Wrote {args.out_json}")

