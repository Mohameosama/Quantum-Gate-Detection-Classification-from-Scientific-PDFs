from __future__ import annotations

import re
import bisect
import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Iterable

import fitz  # PyMuPDF
from PIL import Image

# Optional NLP: spaCy sentencizer (no model download required).
try:  # pragma: no cover
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None

try:
    # When imported as a package module 
    from .extract_figures import ExtractedFigure
except Exception:
    # When executed from repo root / arbitrary CWD
    from pipeline.extract_figures import ExtractedFigure


# ============================================================
# 0) Canonical paper text + spans
# ============================================================
_WORD_RE = re.compile(r"\b\w+\b")
# Only remove true subfigure markers like "(a)" or "a)" at the start.
_SUBFIG_RE = re.compile(r"(?i)^\s*(?:\(\s*[a-z]\s*\)|[a-z]\s*\))\s*")
_FIG_PREFIX_RE = re.compile(r"(?i)^\s*(fig\.?|figure)\s*\d+\s*[:.\-\)]\s*")
_INLINE_SUBFIG_MARK_RE = re.compile(r"(?i)\(\s*[a-z]\s*\)")
_TRAILING_FIG_RE = re.compile(r"(?i)\b(fig\.?|figure)\s*\d+\s*\.?\s*$")


def _tokenize_words(text: str) -> List[str]:
    """Tokenize into simple word tokens for indexing/anchors."""
    return _WORD_RE.findall(text or "")

def _clean_text(s: Optional[str]) -> str:
    """
    Normalize noisy PDF-extracted text:
    - remove leading subfigure markers like '(a)' '(b)'
    - remove inline subfigure markers like '(a)' '(b)' that appear in captions
    - remove trailing "Fig N." fragments that often get duplicated
    - remove repeated whitespace/newlines
    """
    t = (s or "").replace("\u00ad", "")  # soft hyphen
    t = re.sub(r"\s+", " ", t).strip()
    t = _SUBFIG_RE.sub("", t).strip()
    t = _INLINE_SUBFIG_MARK_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _TRAILING_FIG_RE.sub("", t).strip()
    return t

def _clean_caption(s: Optional[str]) -> str:
    """
    Caption-specific cleanup: remove leading 'Fig N.' prefix after we've already extracted fig_no.
    """
    t = _clean_text(s)
    t = _FIG_PREFIX_RE.sub("", t).strip()
    return t

def _normalize_for_dedup(s: str) -> str:
    """Normalize a sentence for similarity comparisons."""
    s = (s or "").lower().strip()
    # keep only letters/numbers/spaces
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _dedup_similar(sentences: List[str], threshold: float = 0.92) -> List[str]:
    """
    Remove near-duplicate sentences (e.g., OCR/PDF extraction artifacts),
    using substring + SequenceMatcher similarity.
    """
    kept: List[str] = []
    kept_norm: List[str] = []

    for s in sentences:
        s = _clean_text(s)
        if not s:
            continue
        n = _normalize_for_dedup(s)
        if not n or len(n) < 20:
            continue

        is_dup = False
        for kn, ks in zip(kept_norm, kept):
            # substring check catches cases like missing first letter ("nder" vs "Under")
            if n in kn or kn in n:
                is_dup = True
                # prefer the longer/more complete string
                if len(s) > len(ks):
                    kept[kept.index(ks)] = s
                    kept_norm[kept_norm.index(kn)] = n
                break
            ratio = difflib.SequenceMatcher(None, n, kn).ratio()
            if ratio >= threshold:
                is_dup = True
                if len(s) > len(ks):
                    kept[kept.index(ks)] = s
                    kept_norm[kept_norm.index(kn)] = n
                break

        if not is_dup:
            kept.append(s)
            kept_norm.append(n)

    return kept


def build_paper_pages(doc: fitz.Document) -> List[str]:
    """Return extracted text per page (1-based pages in doc order)."""
    pages: List[str] = []
    for i in range(len(doc)):
        try:
            pages.append(doc[i].get_text("text") or "")
        except Exception:
            # Some PDFs trigger MuPDF "exception stack overflow" on text extraction.
            # Fail-soft: keep empty page text so the pipeline can continue.
            pages.append("")
    return pages


def build_paper_text(pages: List[str]) -> str:
    """
    Build a canonical paper text with explicit page separators.
    This helps reproducible lookup and debugging.
    """
    parts: List[str] = []
    for i, t in enumerate(pages, 1):
        parts.append(f"\n\n===== PAGE {i} =====\n\n")
        parts.append(t)
    return "".join(parts)


@dataclass
class WordIndex:
    """
    Word-level index of the canonical paper text.

    Attributes
    ----------
    text:
        Canonical paper text.
    word_spans:
        List of (start_char, end_char) for each word token in `text`.
    page_word_offsets:
        Cumulative starting word index per page (len = num_pages + 1).
        For page i (1-based), words are in [offsets[i-1], offsets[i]).
    """
    text: str
    word_spans: List[Tuple[int, int]]
    page_word_offsets: List[int]

    def _word_starts(self) -> List[int]:
        return [a for a, _ in self.word_spans]

    def char_to_word_index(self, char_pos: int) -> int:
        """Return the index of the word at/after char_pos (best effort)."""
        starts = self._word_starts()
        i = bisect.bisect_right(starts, max(0, char_pos)) - 1
        return max(0, min(i, len(self.word_spans) - 1)) if self.word_spans else 0

    def word_index_to_page(self, widx: int) -> int:
        """Map global word index to 1-based page number."""
        # page_word_offsets is sorted; find rightmost offset <= widx
        p = bisect.bisect_right(self.page_word_offsets, max(0, widx))  # 1..num_pages+1
        return max(1, min(p, len(self.page_word_offsets) - 1))


def build_word_index(pages: List[str]) -> WordIndex:
    """
    Build a word index over the canonical paper text, with page->word offsets.
    """
    page_word_offsets: List[int] = [0]
    total = 0
    for t in pages:
        total += len(_tokenize_words(t))
        page_word_offsets.append(total)

    canonical = build_paper_text(pages)

    # Build word spans on canonical text
    spans: List[Tuple[int, int]] = [(m.start(), m.end()) for m in _WORD_RE.finditer(canonical)]
    return WordIndex(text=canonical, word_spans=spans, page_word_offsets=page_word_offsets)

def find_spans(paper_text: str, snippets: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for s in snippets:
        if not s:
            spans.append((-1, -1))
            continue
        i = paper_text.find(s)
        spans.append((i, i + len(s)) if i != -1 else (-1, -1))
    return spans


# ============================================================
# 1) Figure grouping (page-level, geometry-only)
# ============================================================
def _rect_union(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def _rect_center(r: Tuple[float,float,float,float]) -> Tuple[float,float]:
    return ((r[0]+r[2])/2.0, (r[1]+r[3])/2.0)

def _overlap_ratio_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return 0.0 if union <= 0 else inter / union

def should_merge(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    x_ov = _overlap_ratio_1d(ax0, ax1, bx0, bx1)
    y_ov = _overlap_ratio_1d(ay0, ay1, by0, by1)

    x_gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    y_gap = max(0.0, max(ay0, by0) - min(ay1, by1))

    (acx, acy) = _rect_center(a)
    (bcx, bcy) = _rect_center(b)
    dist = ((acx-bcx)**2 + (acy-bcy)**2) ** 0.5

    # We intentionally keep merging conservative:
    # merging two separate figures on the same page is worse than splitting a multi-image figure.
    return (
        (x_ov >= 0.18 and y_gap <= 110.0) or
        (y_ov >= 0.18 and x_gap <= 110.0)
    )

def group_figures_on_page(figs: List[ExtractedFigure]) -> List[Dict[str, Any]]:
    items = [f for f in figs if f.bbox is not None]
    items.sort(key=lambda f: (f.bbox[1], f.bbox[0]))

    groups: List[Dict[str, Any]] = []
    for f in items:
        placed = False
        for g in groups:
            if should_merge(g["bbox"], f.bbox):
                g["members"].append(f)
                g["bbox"] = _rect_union(g["bbox"], f.bbox)
                placed = True
                break
        if not placed:
            groups.append({"bbox": f.bbox, "members": [f]})

    changed = True
    while changed:
        changed = False
        merged: List[Dict[str, Any]] = []
        while groups:
            g = groups.pop(0)
            i = 0
            while i < len(groups):
                if should_merge(g["bbox"], groups[i]["bbox"]):
                    g["bbox"] = _rect_union(g["bbox"], groups[i]["bbox"])
                    g["members"].extend(groups[i]["members"])
                    groups.pop(i)
                    changed = True
                else:
                    i += 1
            merged.append(g)
        groups = merged

    groups.sort(key=lambda g: (g["bbox"][1], g["bbox"][0]))
    return groups


# ============================================================
# 2) Caption linking + confidence
# ============================================================
CAPTION_START_RE = re.compile(
    r"(?im)^\s*(?:\(?[a-z]\)?\s*)?(fig\.?|figure)\s*(\d+)\s*([:.\-\)])?"
)
FIG_ANY_RE = re.compile(r"(?i)\b(fig\.?|figure)\s*(\d+)\b")
FIG_REF_ANY_RE = re.compile(r"(?i)\b(fig\.?|figure)\s*\d+\b")

def _get_blocks(page: fitz.Page):
    blocks = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), (b[4] or "").strip()
        if txt:
            blocks.append((x0, y0, x1, y1, txt))
    return blocks

def _x_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return 0.0 if union <= 0 else inter / union

def link_caption_by_group_bbox(page: fitz.Page, bbox: Tuple[float,float,float,float], max_gap: float = 1200.0):
    ix0, iy0, ix1, iy1 = bbox
    # IMPORTANT: Only search captions in a local region around the figure bbox.
    # This avoids accidentally linking to a different figure caption elsewhere on the page.
    # PDF coords: y increases downward in PyMuPDF.
    clip = fitz.Rect(ix0 - 30, iy0 - 120, ix1 + 30, iy1 + 350)
    blocks = []
    for b in page.get_text("blocks", clip=clip):
        x0, y0, x1, y1, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), (b[4] or "").strip()
        if txt:
            blocks.append((x0, y0, x1, y1, txt))

    candidates_primary: List[Tuple[float, Tuple[float, float, float, float], str, Optional[int], float, float]] = []
    candidates_fallback: List[Tuple[float, Tuple[float, float, float, float], str, Optional[int], float, float]] = []
    for x0, y0, x1, y1, txt in blocks:
        txt_norm = _clean_text(txt)
        # Prefer true caption blocks that START with "Fig/Figure N"
        m = CAPTION_START_RE.search(txt_norm)
        fig_no = int(m.group(2)) if m else None
        if fig_no is None:
            continue
        dist = y0 - iy1
        # allow captions slightly above the figure (some layouts do this)
        if dist < -120.0 or dist > max_gap:
            continue
        xo = _x_overlap(ix0, ix1, x0, x1)
        # scoring: prefer below, but allow above; prefer x-overlap; prefer captions that START with "Fig/Figure"
        starts_like_caption = 1.0 if CAPTION_START_RE.search(txt_norm) else 0.0
        score = (-abs(dist)) + 10.0 * xo + 3.0 * starts_like_caption
        item = (score, (x0, y0, x1, y1), txt_norm.strip(), fig_no, dist, xo)
        # Primary window: near the figure bottom (most likely the correct caption)
        if -40.0 <= dist <= 260.0:
            candidates_primary.append(item)
        else:
            candidates_fallback.append(item)

    candidates = candidates_primary if candidates_primary else (candidates_primary + candidates_fallback)

    if not candidates:
        # Fallback: allow any mention of "Fig N" but only if the block is very close to the figure bbox
        # and not a long body paragraph.
        for x0, y0, x1, y1, txt in blocks:
            txt_norm = _clean_text(txt)
            m2 = FIG_ANY_RE.search(txt_norm)
            fig_no = int(m2.group(2)) if m2 else None
            if fig_no is None:
                continue
            dist = y0 - iy1
            if dist < -60.0 or dist > 260.0:
                continue
            xo = _x_overlap(ix0, ix1, x0, x1)
            if xo < 0.10:
                continue
            # reject long-ish paragraphs (common for in-body "see Fig.2" mentions)
            if len(txt_norm) > 240:
                continue
            score = (-abs(dist)) + 8.0 * xo
            candidates.append((score, (x0, y0, x1, y1), txt_norm.strip(), fig_no, dist, xo))
        if not candidates:
            return None, None, None, False

    candidates.sort(reverse=True)
    cap_bbox, cap_txt, fig_no, dist, xo = candidates[0][1], candidates[0][2], candidates[0][3], candidates[0][4], candidates[0][5]

    # Confidence guard: if the best caption is too far / too weak overlap, better return None
    # than assign a wrong figure number.
    if xo < 0.12 or abs(dist) > 320.0:
        return None, None, None, False

    return fig_no, cap_txt, cap_bbox, True


# ============================================================
# 3) Descriptions = caption + Fig-N reference sentences
# ============================================================
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')

def split_sentences(text: str) -> List[str]:
    """
    Sentence splitting with optional spaCy sentencizer.
    Falls back to a regex splitter if spaCy isn't available.
    """
    t = _clean_text(text)
    if not t:
        return []

    if spacy is not None:
        try:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            doc = nlp(t)
            sents = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
        except Exception:
            sents = [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]
    else:
        sents = [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]

    # filter too-short fragments
    return [s for s in sents if len(s) >= 30]

_DESC_STOP_RE = re.compile(r"(?i)\barxiv\b|copyright|all rights reserved|preprint|doi\b")
_DESC_KEY_TERMS = {
    "quantum", "circuit", "quantum circuit", "gate", "gates", "qubit", "qubits",
    "measurement", "measurements", "register", "ansatz", "variational",
    "algorithm", "oracle", "amplitude", "estimation", "amplification",
}


def _norm_tokens(text: str) -> List[str]:
    return [t.lower() for t in _tokenize_words(text)]


def _caption_tokens(caption: Optional[str]) -> set:
    toks = _norm_tokens(caption or "")
    # remove very short tokens
    return {t for t in toks if len(t) >= 3}


def _score_sentence(sent: str, fig_no: Optional[int], caption_tok: set) -> float:
    s = sent.strip()
    if not s:
        return -1e9
    low = s.lower()
    if _DESC_STOP_RE.search(low):
        return -5.0

    score = 0.0
    if fig_no is not None and re.search(rf"(?i)\b(fig\.?|figure)\s*{fig_no}\b", s):
        score += 3.0
    if fig_no is None and FIG_REF_ANY_RE.search(s):
        # If we don't know the figure number, avoid stealing sentences about other figures.
        score -= 2.0
    if fig_no is not None:
        # If the sentence references a different figure number, penalize heavily.
        for m in re.finditer(r"(?i)\b(fig\.?|figure)\s*(\d+)\b", s):
            try:
                other = int(m.group(2))
            except Exception:
                continue
            if other != fig_no:
                score -= 3.0
                break
    # keyword hits
    for kw in _DESC_KEY_TERMS:
        if kw in low:
            score += 0.6
    # overlap with caption
    stoks = {t for t in _norm_tokens(s) if len(t) >= 3}
    if caption_tok and stoks:
        overlap = len(stoks & caption_tok) / max(1, len(caption_tok))
        score += 2.0 * overlap

    # penalize weirdly short/long
    if len(s) < 40:
        score -= 0.7
    if len(s) > 260:
        score -= 0.5
    return score


def rank_description_sentences(doc: fitz.Document, page_no: int, fig_no: Optional[int], caption_text: Optional[str], max_sentences: int = 3) -> List[str]:
    """
    Rank candidate sentences from pages (page-1, page, page+1) and return top-N.
    Deterministic, no external NLP model required.
    """
    caption_tok = _caption_tokens(_clean_caption(caption_text))
    candidates: List[Tuple[float, int, str]] = []
    for pn in [page_no, page_no - 1, page_no + 1]:
        if pn < 1 or pn > len(doc):
            continue
        try:
            text = doc[pn - 1].get_text("text") or ""
        except Exception:
            continue
        for s in split_sentences(text):
            sc = _score_sentence(s, fig_no, caption_tok)
            # prefer same-page by small tie-break
            dist = abs(pn - page_no)
            candidates.append((sc - 0.15 * dist, dist, s))

    # sort by score desc, then closer pages
    candidates.sort(key=lambda x: (-x[0], x[1], len(x[2])))
    out: List[str] = []
    seen = set()
    for _, _, s in candidates:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_sentences:
            break
    return out

def _local_text_near_bbox(page: fitz.Page, bbox: Tuple[float, float, float, float]) -> str:
    """
    Extract nearby text around a figure bbox from the same page.
    This dramatically reduces cross-figure caption bleed.
    """
    ix0, iy0, ix1, iy1 = bbox
    # Focus on caption + nearby paragraphs: slightly above, and a chunk below.
    clip = fitz.Rect(ix0 - 60, iy0 - 160, ix1 + 60, iy1 + 900)
    try:
        return page.get_text("text", clip=clip) or ""
    except Exception:
        return ""

def rank_description_sentences_local(
    page: fitz.Page,
    bbox: Tuple[float, float, float, float],
    fig_no: Optional[int],
    caption_text: Optional[str],
    max_sentences: int = 3,
) -> List[str]:
    """
    Rank candidate sentences from local clipped text around the figure bbox.
    Falls back to empty if no good candidates are found.
    """
    caption_tok = _caption_tokens(_clean_caption(caption_text))
    text = _local_text_near_bbox(page, bbox)
    sents = split_sentences(text)
    scored: List[Tuple[float, str]] = []
    for s in sents:
        sc = _score_sentence(s, fig_no, caption_tok)
        # If we know fig_no, prefer sentences that actually reference it or overlap with caption.
        if fig_no is not None:
            has_ref = bool(re.search(rf"(?i)\b(fig\.?|figure)\s*{fig_no}\b", s))
            if not has_ref and caption_tok:
                stoks = {t for t in _norm_tokens(s) if len(t) >= 3}
                overlap = len(stoks & caption_tok) / max(1, len(caption_tok))
                if overlap < 0.12:
                    sc -= 1.5
        scored.append((sc, s))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    out: List[str] = []
    for sc, s in scored:
        if sc < 0.2:
            continue
        out.append(s)
        if len(out) >= max_sentences:
            break
    return out

def fig_reference_sentences(doc: fitz.Document, page_no: int, fig_no: int, max_sentences: int = 2) -> List[str]:
    pat = re.compile(rf'(?i)\b(fig\.?|figure)\s*{fig_no}\b')
    candidates: List[Tuple[int, str]] = []
    for pn in [page_no-1, page_no, page_no+1]:
        if pn < 1 or pn > len(doc):
            continue
        t = doc[pn-1].get_text("text") or ""
        for s in split_sentences(t):
            if pat.search(s):
                candidates.append((abs(pn - page_no), s))
    seen = set()
    uniq = []
    for d, s in candidates:
        if s in seen:
            continue
        seen.add(s)
        uniq.append((d, s))
    uniq.sort(key=lambda x: (x[0], len(x[1])))
    return [s for _, s in uniq[:max_sentences]]

def build_descriptions(doc: fitz.Document, page: fitz.Page, bbox: Tuple[float, float, float, float], page_no: int, fig_no: Optional[int], caption_text: Optional[str]) -> Tuple[List[str], str]:
    desc: List[str] = []
    if caption_text:
        desc.append(_clean_caption(caption_text))
    # 1) Prefer local context to avoid picking captions/text from other figures on the same page.
    ranked_local = rank_description_sentences_local(page, bbox, fig_no, caption_text, max_sentences=3)
    desc.extend([_clean_text(s) for s in ranked_local])
    # 2) If local context is sparse, fall back to broader page-based ranking.
    if len(desc) < 2:
        ranked = rank_description_sentences(doc, page_no, fig_no, caption_text, max_sentences=3)
        desc.extend([_clean_text(s) for s in ranked])
    desc = _dedup_similar(desc, threshold=0.92)
    return desc, "caption+local_ranked_sentences+dedup"


# ============================================================
# 4) Render helper (used by pipeline to save the grouped figure image)
# ============================================================
def render_bbox(page: fitz.Page, bbox: Tuple[float,float,float,float], dpi: int = 520) -> Image.Image:
    # Normalize bbox then clamp to page bounds to avoid invalid clips that can produce corrupted pixmaps.
    x0, y0, x1, y1 = bbox
    nx0, nx1 = (x0, x1) if x0 <= x1 else (x1, x0)
    ny0, ny1 = (y0, y1) if y0 <= y1 else (y1, y0)
    rect = fitz.Rect(float(nx0), float(ny0), float(nx1), float(ny1))
    rect = rect & page.rect
    if rect.is_empty or rect.width <= 1 or rect.height <= 1:
        raise ValueError(f"Empty/invalid render rect after clamping: {rect}")
    pix = page.get_pixmap(clip=rect, dpi=dpi, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ============================================================
# 5) Quantum problem inference (evidence only) + audit
# ============================================================
HEADING_RE = re.compile(r"(?m)^\s*(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])\.?\s+[A-Za-z].{3,160}$")

def extract_headings_with_y(page: fitz.Page) -> List[Tuple[float, str]]:
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
    out.sort(key=lambda t: t[0])
    return out

def infer_problem(evidence: List[str], heading: Optional[str]) -> Tuple[str, str]:
    text = "\n".join(evidence).lower()
    h = (heading or "").lower()

    def has_any(src: str, keys: List[str]) -> bool:
        return any(k in src for k in keys)

    # Search heading first (often cleaner), then body text.
    for method, src in (("heading", h), ("evidence", text)):

        # -----------------------------
        # Quantum error correction / FTQC
        # -----------------------------
        if has_any(src, [
            "error correction", "quantum error correction", "qec", "fault-tolerant", "fault tolerant",
            "surface code", "toric code", "color code", "stabilizer", "syndrome", "decoder",
            "logical qubit", "magic state", "distillation", "lattice surgery",
        ]):
            return "Quantum error correction / fault tolerance", method

        # -----------------------------
        # Cryptography / comms
        # -----------------------------
        if has_any(src, ["qkd", "quantum key distribution", "bb84", "e91", "device-independent qkd", "diqkd"]):
            return "Quantum key distribution (QKD)", method
        if has_any(src, ["quantum teleportation", "teleportation protocol"]):
            return "Quantum teleportation", method
        if has_any(src, ["superdense coding", "dense coding"]):
            return "Superdense coding", method
        if has_any(src, ["entanglement swapping", "entanglement distribution", "quantum repeater"]):
            return "Entanglement distribution / repeater", method

        # -----------------------------
        # Core named algorithms
        # -----------------------------
        if has_any(src, ["shor", "order finding", "factoring"]):
            return "Shor's algorithm / factoring", method
        if has_any(src, ["grover", "amplitude amplification", "unstructured search"]):
            return "Grover search / amplitude amplification", method
        if has_any(src, ["quantum walk", "walk-based", "discrete-time quantum walk", "continuous-time quantum walk"]):
            return "Quantum walk algorithms", method
        if has_any(src, ["hhl", "harrow hassidim lloyd", "linear system", "quantum linear systems algorithm", "qlsa"]):
            return "Quantum linear systems (HHL/QLSA)", method

        # -----------------------------
        # Phase estimation / QFT
        # -----------------------------
        if has_any(src, ["phase estimation", "quantum phase estimation", "qpe", "iterative phase estimation"]):
            return "Quantum phase estimation (QPE)", method
        if has_any(src, ["fourier transform", "quantum fourier", "qft"]):
            return "Quantum Fourier Transform (QFT)", method

        # -----------------------------
        # Amplitude estimation / Monte Carlo
        # -----------------------------
        if has_any(src, [
            "amplitude estimation", "quantum amplitude estimation", "qae",
            "quantum monte carlo", "monte carlo", "expectation estimation", "sampling-based estimation",
        ]):
            return "Amplitude estimation / quantum Monte Carlo", method

        # -----------------------------
        # Optimization (variational + non-variational)
        # -----------------------------
        if has_any(src, ["qaoa", "quantum approximate optimization algorithm"]):
            return "QAOA", method
        if has_any(src, [
            "vqe", "variational quantum eigensolver", "ground state", "eigensolver", "excited state",
            "ucco", "uccsd", "unitary coupled cluster",
        ]):
            return "VQE / variational eigensolver", method
        if has_any(src, [
            "variational", "ansatz", "parameterized quantum circuit", "pqc",
            "variational circuit", "variational form",
        ]):
            return "Variational quantum algorithm (general)", method

        # -----------------------------
        # Quantum machine learning
        # -----------------------------
        if has_any(src, [
            "quantum neural network", "qnn", "variational classifier", "quantum classifier",
            "quantum machine learning", "qml", "vqcnn", "vqc", "quantum kernel",
            "kernel method", "support vector", "svm", "quantum feature map", "feature map",
            "quantum convolution", "quantum perceptron",
            # training-ish terms frequently used in QML/VQC papers
            "training", "optimizer", "loss", "gradient", "backprop", "epochs", "dataset",
        ]):
            return "Quantum machine learning", method
        if has_any(src, ["quantum kernel", "kernel estimation", "fidelity kernel", "projected kernel"]):
            return "Quantum kernel methods", method
        if has_any(src, [
            "quantum gan", "qgan", "generative adversarial", "born machine", "qcbm",
            "quantum generative model",
        ]):
            return "Quantum generative modeling", method

        # -----------------------------
        # Simulation / chemistry / physics
        # -----------------------------
        if has_any(src, [
            "hamiltonian simulation", "trotter", "trotterization", "lie-trotter", "suzuki",
            "time evolution", "unitary evolution", "digital simulation",
        ]):
            return "Hamiltonian simulation / time evolution", method
        if has_any(src, [
            "quantum chemistry", "molecular", "hartree-fock", "fermionic", "bravyi-kitaev", "jordan-wigner",
            "electronic structure",
        ]):
            return "Quantum chemistry / electronic structure", method
        if has_any(src, [
            "ising", "max-cut", "maxcut", "qubo", "combinatorial optimization", "tsp", "traveling salesman",
            "portfolio optimization", "facility location", "graph coloring",
        ]):
            return "Combinatorial optimization (Ising/QUBO)", method

        # -----------------------------
        # Finance (more specific)
        # -----------------------------
        if has_any(src, ["option pricing", "payoff", "strike", "black-scholes", "black scholes", "financial risk", "var", "cvar"]):
            return "Quantum finance (option pricing / risk)", method

        # -----------------------------
        # State preparation / encoding / tomography
        # -----------------------------
        if has_any(src, ["state preparation", "prepare the state", "encoding", "data encoding", "feature encoding", "amplitude encoding"]):
            return "State preparation / encoding", method
        if has_any(src, ["tomography", "state tomography", "process tomography", "quantum process tomography"]):
            return "Quantum tomography / characterization", method

        # -----------------------------
        # Benchmarking / characterization / noise
        # -----------------------------
        if has_any(src, [
            "randomized benchmarking", "rb", "cycle benchmarking", "interleaved benchmarking",
            "noise characterization", "noise model", "depolarizing", "readout error", "crosstalk",
            "calibration", "t1", "t2", "coherence time",
        ]):
            return "Benchmarking / noise characterization", method

        # -----------------------------
        # Compilation / transpilation / mapping
        # -----------------------------
        if has_any(src, [
            "transpile", "transpilation", "compiler", "compilation", "routing", "mapping", "layout",
            "qubit mapping", "swap insertion", "gate decomposition", "synthesis",
            "optimization pass", "circuit optimization",
        ]):
            return "Quantum compilation / circuit optimization", method

        # -----------------------------
        # Distributed / networking / modular computing
        # -----------------------------
        if has_any(src, ["distributed quantum computing", "modular", "networked quantum", "quantum network"]):
            return "Distributed / networked quantum computing", method

        # -----------------------------
        # Randomness / sampling
        # -----------------------------
        if has_any(src, ["random injection"]) or (("random" in src) and ("injection" in src)):
            return "Random injection / randomness generation", method
        if has_any(src, [
            "random number", "rng", "quantum random", "randomness", "sampling task",
            "boson sampling", "random circuit sampling", "rcs",
        ]):
            return "Randomness generation / sampling", method

        # -----------------------------
        # Measurement / readout / mitigation
        # -----------------------------
        if has_any(src, [
            "error mitigation", "zne", "zero-noise extrapolation", "readout mitigation",
            "mitigation", "probabilistic error cancellation", "pec",
        ]):
            return "Error mitigation", method
        if has_any(src, ["measurement", "readout", "measurement protocol", "mid-circuit measurement"]):
            return "Measurement / readout", method

        # -----------------------------
        # Control / pulses / hardware (very generic; keep last)
        # -----------------------------
        if has_any(src, [
            "pulse", "pulse-level", "optimal control", "grape", "crab", "dragg", "microwave pulse",
            "control sequence", "control electronics",
        ]):
            return "Quantum control / pulse shaping", method
        if has_any(src, [
            "superconducting", "transmon", "ion trap", "trapped ion", "photonic", "nv center",
            "spin qubit", "quantum dot",
        ]):
            return "Quantum hardware / device architecture", method

    return "", "none"


# ============================================================
# 6) Context + Enrich one figure group (audit fields)
# ============================================================
@dataclass
class PDFContext:
    doc: fitz.Document
    pages_text: List[str]
    paper_text: str
    word_index: WordIndex

    @classmethod
    def open(cls, pdf_path: str) -> "PDFContext":
        doc = fitz.open(pdf_path)
        # IMPORTANT: Some PDFs crash MuPDF during text extraction (e.g., "exception stack overflow").
        # We keep this fail-soft: if building a full text index fails, continue with empty text.
        try:
            pages = build_paper_pages(doc)
            widx = build_word_index(pages)
            return cls(doc=doc, pages_text=pages, paper_text=widx.text, word_index=widx)
        except Exception:
            empty_offsets = [0] * (len(doc) + 1)
            widx = WordIndex(text="", word_spans=[], page_word_offsets=empty_offsets)
            return cls(doc=doc, pages_text=[], paper_text="", word_index=widx)

    def close(self) -> None:
        self.doc.close()


def enrich_figure_group(
    fig_group_bbox: Tuple[float,float,float,float],
    members: List[ExtractedFigure],
    pdf_path: str,
    ctx: PDFContext,
) -> Dict[str, Any]:
    page_number = members[0].page_number
    page = ctx.doc[page_number - 1]

    fig_no, caption, cap_bbox, cap_ok = link_caption_by_group_bbox(page, fig_group_bbox)
    descriptions, desc_method = build_descriptions(ctx.doc, page, fig_group_bbox, page_number, fig_no, caption)
    # Word-anchored positions (easy to read and stable)
    text_positions = []
    for d in descriptions:
        words = [w for w in _tokenize_words(d) if len(w) > 1]
        n = 6
        start_words = " ".join(words[:n]) if words else ""
        end_words = " ".join(words[-n:]) if words else ""
        text_positions.append({"start_words": start_words, "end_words": end_words})

    headings = extract_headings_with_y(page)
    anchor_y = fig_group_bbox[1]
    if cap_bbox:
        anchor_y = min(anchor_y, cap_bbox[1])
    heading = None
    for y, htxt in headings:
        if y <= anchor_y + 1e-6:
            heading = htxt
        else:
            break
    problem, prob_method = infer_problem(descriptions, heading)

    return {
        "arxiv_number": members[0].pdf_name,
        "page_number": int(page_number),
        "figure_number": int(fig_no) if fig_no is not None else None,

        "quantum_problem": problem,
        "descriptions": descriptions,
        "text_positions": text_positions,
        "member_images": [str(m.file_path) for m in members],

        "_caption_confidence": "high" if cap_ok else "none",
        "_description_method": desc_method,
        "_problem_method": prob_method,
        "_heading": heading,
        "_group_bbox": tuple(float(x) for x in fig_group_bbox),
    }
