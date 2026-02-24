# enrich_extracted_figure_v2.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

try:
    from pipeline.extract_figures import ExtractedFigure
except Exception:
    # Keep archive scripts runnable from within archive/ as well
    from ..pipeline.extract_figures import ExtractedFigure


def build_paper_text(doc: fitz.Document) -> str:
    parts: List[str] = []
    for i in range(len(doc)):
        parts.append(f"\n\n===== PAGE {i+1} =====\n\n")
        parts.append(doc[i].get_text("text") or "")
    return "".join(parts)


def find_spans(paper_text: str, snippets: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for s in snippets:
        if not s:
            spans.append((-1, -1))
            continue
        i = paper_text.find(s)
        spans.append((i, i + len(s)) if i != -1 else (-1, -1))
    return spans


CAPTION_START_RE = re.compile(r'(?im)^\s*(fig\.?|figure)\s*(\d+)\s*[:.\-]')


def _x_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return 0.0 if union <= 0 else inter / union


def _get_blocks(page: fitz.Page):
    blocks = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), (b[4] or "").strip()
        if txt:
            blocks.append((x0, y0, x1, y1, txt))
    return blocks


def link_caption_by_bbox(
    page: fitz.Page,
    bbox: Tuple[float, float, float, float],
    max_gap: float = 220.0
) -> Tuple[Optional[int], Optional[str], Optional[Tuple[float, float, float, float]]]:
    ix0, iy0, ix1, iy1 = bbox
    blocks = _get_blocks(page)

    below = []
    above = []

    for x0, y0, x1, y1, txt in blocks:
        if not CAPTION_START_RE.search(txt):
            continue

        dist_below = y0 - iy1
        dist_above = iy0 - y1
        xo = _x_overlap(ix0, ix1, x0, x1)

        if dist_below >= -3 and dist_below <= max_gap:
            score = -dist_below + 10.0 * xo
            below.append((score, (x0, y0, x1, y1), txt))
        elif dist_above >= 0 and dist_above <= max_gap:
            score = -dist_above + 10.0 * xo
            above.append((score, (x0, y0, x1, y1), txt))

    if below:
        below.sort(reverse=True)
        cap_bbox, cap_txt = below[0][1], below[0][2]
    elif above:
        above.sort(reverse=True)
        cap_bbox, cap_txt = above[0][1], above[0][2]
    else:
        return None, None, None

    m = CAPTION_START_RE.search(cap_txt)
    fig_no = int(m.group(2)) if m else None
    return fig_no, cap_txt.strip(), cap_bbox


def render_bbox(page: fitz.Page, bbox: Tuple[float, float, float, float], dpi: int = 300) -> Image.Image:
    rect = fitz.Rect(*bbox)
    pix = page.get_pixmap(clip=rect, dpi=dpi, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


_GATE_CANON = {
    "H": "h", "X": "x", "Y": "y", "Z": "z",
    "S": "s", "T": "t",
    "RX": "rx", "RY": "ry", "RZ": "rz",
    "U1": "u1", "U2": "u2", "U3": "u3",
    "CX": "cx", "CNOT": "cx",
    "CZ": "cz",
    "CCX": "ccx", "CCNOT": "ccx", "TOFFOLI": "ccx",
    "SWAP": "swap", "ISWAP": "iswap",
    "MEASURE": "measure", "MEAS": "measure", "M": "measure",
}

_PARAM_PREFIX = re.compile(r"^(RX|RY|RZ|U1|U2|U3|CRX|CRY|CRZ)\b")
_WORD_CLEAN = re.compile(r"[^A-Z0-9]+")


def _normalize_gate_token(tok: str) -> Optional[str]:
    t = tok.strip().upper()
    if not t:
        return None
    t = _WORD_CLEAN.sub("", t)

    if t in {"CONOT", "CCNOT"}:
        t = "CCNOT"
    if t in {"CNO", "CN0"}:
        t = "CNOT"

    m = _PARAM_PREFIX.match(t)
    if m:
        t = m.group(1)

    return _GATE_CANON.get(t)


def extract_gates_from_ocr(img: Image.Image) -> List[str]:
    config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()/-"
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)

    found = set()
    for txt in data.get("text", []):
        txt = (txt or "").strip()
        if not txt:
            continue
        g = _normalize_gate_token(txt)
        if g:
            found.add(g)

    return sorted(found)


HEADING_RE = re.compile(r"(?m)^\s*\d+(\.\d+)*\s+[A-Z].{3,80}$")


def _extract_headings_with_y(page: fitz.Page) -> List[Tuple[float, str]]:
    d = page.get_text("dict")
    out: List[Tuple[float, str]] = []
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = (text or "").strip()
            if not text:
                continue
            if HEADING_RE.match(text):
                y = float(line["bbox"][1])
                out.append((y, text))
    out.sort(key=lambda x: x[0])
    return out


def infer_problem(descriptions: List[str], heading: str | None) -> str:
    text = "\n".join(descriptions).lower()
    h = (heading or "").lower()

    for src in (h, text):
        if any(k in src for k in ["payoff", "strike", "option", "pricing", "financial"]):
            return "Quantum payoff computation / option pricing"
        if "random injection" in src or ("random" in src and "injection" in src):
            return "Random injection / randomness generation"
        if "error correction" in src or "surface code" in src or "stabilizer" in src:
            return "Quantum error correction"
        if "variational" in src or "vqe" in src or "ansatz" in src:
            return "Variational quantum algorithm"
        if "qaoa" in src:
            return "QAOA"
        if "phase estimation" in src or "qpe" in src:
            return "Quantum phase estimation"
        if "fourier" in src or "qft" in src:
            return "Quantum Fourier Transform"

    return ""


@dataclass
class PDFContext:
    doc: fitz.Document
    paper_text: str

    @classmethod
    def open(cls, pdf_path: str) -> "PDFContext":
        doc = fitz.open(pdf_path)
        paper_text = build_paper_text(doc)
        return cls(doc=doc, paper_text=paper_text)

    def close(self) -> None:
        self.doc.close()


def enrich_extracted_figure(
    pdf_path: str,
    fig: ExtractedFigure,
    ctx: Optional[PDFContext] = None,
) -> Dict[str, Any]:
    owns_ctx = False
    if ctx is None:
        ctx = PDFContext.open(pdf_path)
        owns_ctx = True

    page = ctx.doc[fig.page_number - 1]
    page_text = page.get_text("text") or ""

    fig_no = None
    caption = None
    cap_bbox = None

    if fig.bbox:
        fig_no, caption, cap_bbox = link_caption_by_bbox(page, fig.bbox)

    descriptions: List[str] = []
    if caption:
        descriptions.append(caption)

    paras = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
    if fig_no is not None:
        pat = re.compile(rf"(?i)\b(fig\.?|figure)\s*{fig_no}\b")
        hits = [p for p in paras if pat.search(p)]
        if hits:
            descriptions.extend(hits[:2])

    if len(descriptions) <= 1:
        KEYWORDS = {"quantum", "circuit", "qubit", "gate", "measurement", "algorithm", "ansatz", "payoff", "pricing"}
        scored = sorted(paras, key=lambda p: sum(k in p.lower() for k in KEYWORDS), reverse=True)
        descriptions.extend(scored[:2])

    seen = set()
    descriptions = [d for d in descriptions if not (d in seen or seen.add(d))]

    text_positions = find_spans(ctx.paper_text, descriptions)

    headings = _extract_headings_with_y(page)
    anchor_y = fig.bbox[1] if fig.bbox else 1e9
    if cap_bbox:
        anchor_y = min(anchor_y, cap_bbox[1])

    heading = None
    for y, htxt in headings:
        if y <= anchor_y + 1e-6:
            heading = htxt
        else:
            break

    gates: List[str] = []
    if fig.bbox:
        img = render_bbox(page, fig.bbox, dpi=320)
        gates = extract_gates_from_ocr(img)

    problem = infer_problem(descriptions, heading)

    out = {
        "arxiv_number": fig.pdf_name,
        "page_number": int(fig.page_number),
        "figure_number": int(fig_no) if fig_no is not None else None,
        "quantum_gates": gates,
        "quantum_problem": problem,
        "descriptions": descriptions,
        "text_positions": [(int(a), int(b)) for a, b in text_positions],
        "_heading": heading,
    }

    if owns_ctx:
        ctx.close()

    return out
