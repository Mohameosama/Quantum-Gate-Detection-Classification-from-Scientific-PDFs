from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure repo root + src/ are on sys.path (stable imports regardless of CWD)
# File location: <repo_root>/src/pipeline/run_full_pipeline_batch.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for p in (str(_SRC_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from .download_papers import load_paper_list, extract_arxiv_id, download_paper
    from .extract_figures import extract_figures_from_pdf, ExtractedFigure
    from .enrich_figure_groups import PDFContext, group_figures_on_page, enrich_figure_group, render_bbox
except Exception:
    from pipeline.download_papers import load_paper_list, extract_arxiv_id, download_paper
    from pipeline.extract_figures import extract_figures_from_pdf, ExtractedFigure
    from pipeline.enrich_figure_groups import PDFContext, group_figures_on_page, enrich_figure_group, render_bbox

from predict.predict_qc_folder import build_model, get_transform, predict_image
from predict.gate_candidate_extractor import save_candidates
from predict.predict_gates_from_qc_image import load_model as load_gate_model, predict_on_crops, nms
from PIL import Image as PILImage


def _default_path(p: str) -> str:
    """Small helper for argparse defaults shown as strings."""
    return str(Path(p))


def _ensure_pdf_cached(arxiv_raw: str, pdf_dir: Path) -> Optional[Path]:
    """
    Ensure the PDF for an arXiv id exists in pdf_dir (download if missing).

    Parameters
    ----------
    arxiv_raw:
        Input id from paper list (e.g. 'arXiv:2507.23310' or '2507.23310')
    pdf_dir:
        Directory where cached PDFs are stored.

    Returns
    -------
    Optional[Path]
        Path to the cached PDF, or None if download failed.
    """
    clean_id = extract_arxiv_id(arxiv_raw)
    pdf_path = pdf_dir / f"{clean_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    out = download_paper(clean_id, str(pdf_dir))
    return Path(out) if out else None

def _write_json_atomic(path: Path, obj: Any) -> None:
    """
    Write JSON atomically to avoid half-written files if the job is interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def _write_counts_csv(path: Path, paper_ids: List[str], counts: Dict[str, Optional[int]]) -> None:
    """
    Write paper_list_counts CSV incrementally. Blank = not processed yet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "extracted_images"])
        for pid in paper_ids:
            c = counts.get(pid, None)
            w.writerow([pid, "" if c is None else int(c)])
    tmp.replace(path)

def _save_png_safe(img: PILImage.Image, out_path: Path) -> None:
    """
    Save PNG robustly. If Pillow hits a low-level encoder issue, retry using a fully
    materialized RGB copy and a BytesIO roundtrip. Raises on final failure.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) direct save
    try:
        img.save(str(out_path), format="PNG")
        return
    except Exception:
        pass

    # 2) force load/copy and save
    try:
        img2 = img.convert("RGB").copy()
        img2.save(str(out_path), format="PNG")
        return
    except Exception:
        pass

    # 3) BytesIO roundtrip
    bio = io.BytesIO()
    img3 = img.convert("RGB").copy()
    img3.save(bio, format="PNG")
    out_path.write_bytes(bio.getvalue())

def _read_counts_csv(path: Path) -> Dict[str, Optional[int]]:
    """
    Read an existing counts CSV.
    Blank/empty values are treated as not processed yet (None).
    """
    if not path.exists():
        return {}
    out: Dict[str, Optional[int]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row:
                continue
            pid = row[0].strip()
            c = row[1].strip() if len(row) > 1 else ""
            if c == "":
                out[pid] = None
            else:
                try:
                    out[pid] = int(c)
                except Exception:
                    out[pid] = None
    return out

def _read_dataset_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # If file is corrupted/partial, don't crash; start fresh.
        return {}


def _make_output_name(arxiv_id: str, page_no: int, fig_no: Optional[int], group_idx: int) -> str:
    """
    Deterministic output filename for an accepted QC figure image.
    We intentionally do NOT include the word 'group' in filenames.
    """
    fig_str = str(fig_no) if fig_no is not None else "NA"
    return f"{arxiv_id}_page{page_no}_fig{fig_str}.png"


def _filter_required_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only the required fields for the final dataset JSON.
    """
    return {
        "arxiv_number": rec.get("arxiv_number", ""),
        "page_number": int(rec.get("page_number", 0)) if rec.get("page_number") is not None else 0,
        "figure_number": rec.get("figure_number", None),
        "quantum_gates": [],
        "quantum_problem": rec.get("quantum_problem", ""),
        "descriptions": rec.get("descriptions", []),
        "text_positions": rec.get("text_positions", []),
    }

def _predict_gates_for_image(
    image_path: Path,
    *,
    gate_model,
    idx_to_label: Dict[int, str],
    tfm,
    device: torch.device,
    tmp_dir: Path,
    pad: int,
    min_conf: float,
    iou: float,
) -> List[str]:
    """
    Predict quantum gates from a QC image using the gate patch classifier only (no OCR).
    """
    cand_dir = tmp_dir / image_path.stem
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand_json = save_candidates(image_path, cand_dir, pad=pad)
    meta = json.loads(Path(cand_json).read_text(encoding="utf-8"))
    candidates = meta.get("candidates", [])

    preds = predict_on_crops(gate_model, idx_to_label, tfm, candidates, device=device)

    # Per-class thresholds (keep consistent with predict_gates_from_qc_image.py)
    strict = {"t": 0.85, "s": 0.85, "z": 0.85, "p": 0.85, "r": 0.85}
    kept = []
    for p in preds:
        lab = p["pred_label"]
        thr = strict.get(lab, min_conf)
        if float(p["confidence"]) >= thr and lab != "other":
            kept.append(p)

    # NMS de-dup per label
    dedup = []
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for k in kept:
        by_label.setdefault(k["pred_label"], []).append(k)
    for _, lst in by_label.items():
        dedup.extend(nms(lst, iou_thr=iou))

    gates = sorted({d["pred_label"] for d in dedup})
    # cleanup per-image candidates to avoid storing lots of intermediate crops
    shutil.rmtree(cand_dir, ignore_errors=True)
    return gates


def run_batch(
    exam_id: str,
    paper_list_file: Path,
    pdf_dir: Path,
    qc_ckpt: Path,
    gate_ckpt: Path,
    images_out_dir: Path,
    dataset_json_path: Path,
    counts_csv_path: Path,
    *,
    qc_threshold: float = 0.8,
    min_size: int = 50,
    max_images: int = 250,
    gate_pad: int = 6,
    gate_min_conf: float = 0.70,
    gate_iou: float = 0.35,
    render_dpi: int = 300,
    resume: bool = True,
    tmp_root: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """
    Batch pipeline:
    - Read paper list in order
    - Ensure PDFs are cached (download missing)
    - Extract figures from PDF, run QC filter, group by page bbox, enrich metadata
    - Render group bbox images into images_<exam_id> until max_images reached
    - Write dataset JSON and paper_list_counts_<exam_id>.csv

    Returns
    -------
    (images_out_dir, dataset_json_path, counts_csv_path)
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)
    dataset_json_path.parent.mkdir(parents=True, exist_ok=True)
    counts_csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = tmp_root or (Path("pipeline_tmp") / f"exam_{exam_id}")
    tmp_root.mkdir(parents=True, exist_ok=True)

    paper_ids = load_paper_list(str(paper_list_file))
    print(f"[INFO] Starting batch pipeline for exam_id={exam_id}", flush=True)
    print(f"[INFO] Papers in list: {len(paper_ids)} | target_images={max_images}", flush=True)
    print(f"[INFO] PDF cache dir: {pdf_dir}", flush=True)
    print(f"[INFO] Output images: {images_out_dir}", flush=True)
    print(f"[INFO] Output dataset: {dataset_json_path}", flush=True)
    print(f"[INFO] Output counts : {counts_csv_path}", flush=True)

    # Load QC classifier
    device = torch.device("cpu")
    ckpt = torch.load(str(qc_ckpt), map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    img_size = ckpt.get("img_size", 224)
    model = build_model(num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    transform = get_transform(int(img_size))

    # Load gate patch model (CNN)
    gate_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate_model, gate_idx_to_label, gate_tfm = load_gate_model(Path(gate_ckpt), gate_device)
    gate_tmp = tmp_root / "gate_candidates"
    gate_tmp.mkdir(parents=True, exist_ok=True)

    # Outputs (support resume)
    dataset: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, Optional[int]] = {pid: None for pid in paper_ids}  # None => not processed

    if resume and (dataset_json_path.exists() or counts_csv_path.exists()):
        dataset = _read_dataset_json(dataset_json_path)
        prev_counts = _read_counts_csv(counts_csv_path)
        for pid in paper_ids:
            if pid in prev_counts:
                counts[pid] = prev_counts[pid]
        print(f"[INFO] Resume enabled: existing dataset items={len(dataset)}", flush=True)
    else:
        # ensure fresh files exist for streaming consumers
        _write_json_atomic(dataset_json_path, dataset)
        _write_counts_csv(counts_csv_path, paper_ids, counts)

    total_images = len(dataset)
    if total_images > 0:
        print(f"[INFO] Resuming with total_images={total_images}/{max_images}", flush=True)

    for paper_idx, pid in enumerate(paper_ids, 1):
        if total_images >= max_images:
            break

        # Skip papers already processed according to counts CSV
        if counts.get(pid, None) is not None:
            continue

        clean_id = extract_arxiv_id(pid)
        expected_pdf = pdf_dir / f"{clean_id}.pdf"
        if expected_pdf.exists():
            print(f"\n[PDF {paper_idx}/{len(paper_ids)}] {pid} -> cached", flush=True)
        else:
            print(f"\n[PDF {paper_idx}/{len(paper_ids)}] {pid} -> downloading...", flush=True)

        t_paper0 = time.time()
        pdf_path = _ensure_pdf_cached(pid, pdf_dir)
        if pdf_path is None or not pdf_path.exists():
            counts[pid] = 0
            _write_counts_csv(counts_csv_path, paper_ids, counts)
            print(f"[WARN] Download failed for {pid}. Marked count=0 and continuing.", flush=True)
            continue

        arxiv_id = pdf_path.stem
        raw_dir = tmp_root / f"images_raw_{arxiv_id}"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Extract figures; handle corrupted PDFs by deleting and re-downloading once.
        extracted: List[ExtractedFigure] = []
        for attempt in range(2):
            try:
                extracted = extract_figures_from_pdf(
                    pdf_path=str(pdf_path),
                    output_dir=str(raw_dir),
                    min_size=min_size,
                )
                break
            except Exception:
                if attempt == 0:
                    # likely corrupted cache: remove and try re-download once
                    print(f"[WARN] PDF read/extract failed for {arxiv_id}. Deleting cache and re-downloading once...", flush=True)
                    try:
                        pdf_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    redl = download_paper(arxiv_id, str(pdf_dir))
                    pdf_path = Path(redl) if redl else pdf_path
                    continue
                # give up for this paper
                extracted = []

        if not extracted:
            counts[pid] = 0
            _write_counts_csv(counts_csv_path, paper_ids, counts)
            shutil.rmtree(raw_dir, ignore_errors=True)
            print(f"[WARN] No extractable figures for {arxiv_id}. Marked count=0 and continuing.", flush=True)
            continue

        print(f"[INFO] Extracted figures from {arxiv_id}: {len(extracted)} (min_size={min_size})", flush=True)

        # QC probabilities per extracted figure
        qc_flags: Dict[str, bool] = {}
        for fig in extracted:
            p = float(predict_image(model, fig.file_path, transform, device))
            qc_flags[fig.file_path] = (p >= qc_threshold)

        by_page: Dict[int, List[ExtractedFigure]] = {}
        for fig in extracted:
            by_page.setdefault(fig.page_number, []).append(fig)

        per_paper_count = 0

        ctx = PDFContext.open(str(pdf_path))
        try:
            for page_no, figs in sorted(by_page.items()):
                if total_images >= max_images:
                    break

                groups = group_figures_on_page(figs)
                for gi, g in enumerate(groups):
                    if total_images >= max_images:
                        break

                    members = g["members"]
                    if not any(qc_flags.get(m.file_path, False) for m in members):
                        continue

                    # Render group bbox and save into final images_<exam_id> folder
                    page = ctx.doc[page_no - 1]
                    try:
                        img = render_bbox(page, g["bbox"], dpi=int(render_dpi))
                    except Exception as e:
                        print(f"[WARN] Failed to render bbox for {arxiv_id} page={page_no}: {e}. Skipping this group.", flush=True)
                        continue

                    rec_full = enrich_figure_group(
                        fig_group_bbox=g["bbox"],
                        members=members,
                        pdf_path=str(pdf_path),
                        ctx=ctx,
                    )

                    out_name = _make_output_name(arxiv_id, page_no, rec_full.get("figure_number"), gi)
                    out_path = images_out_dir / out_name
                    # Avoid collisions deterministically
                    if out_path.exists():
                        fig_str = str(rec_full.get("figure_number")) if rec_full.get("figure_number") is not None else "NA"
                        out_name = f"{arxiv_id}_page{page_no}_fig{fig_str}_part{gi}.png"
                        out_path = images_out_dir / out_name

                    try:
                        _save_png_safe(img, out_path)
                    except Exception as e:
                        print(f"[WARN] Failed to save image {out_path.name}: {e}. Skipping this group.", flush=True)
                        continue

                    rec = _filter_required_fields(rec_full)
                    # Gate detection: model-only (gate patch CNN)
                    rec["quantum_gates"] = _predict_gates_for_image(
                        out_path,
                        gate_model=gate_model,
                        idx_to_label=gate_idx_to_label,
                        tfm=gate_tfm,
                        device=gate_device,
                        tmp_dir=gate_tmp,
                        pad=gate_pad,
                        min_conf=gate_min_conf,
                        iou=gate_iou,
                    )
                    dataset[out_name] = rec
                    total_images += 1
                    per_paper_count += 1
                    _write_json_atomic(dataset_json_path, dataset)
                    print(
                        f"[ACCEPTED {total_images}/{max_images}] {out_name} | "
                        f"gates={len(rec['quantum_gates'])} | paper={arxiv_id} page={page_no} fig={rec.get('figure_number')}",
                        flush=True,
                    )
        finally:
            ctx.close()
            # Remove temporary extracted figures; keep only accepted QC images in images_<exam_id>
            shutil.rmtree(raw_dir, ignore_errors=True)

        counts[pid] = per_paper_count
        _write_counts_csv(counts_csv_path, paper_ids, counts)
        dt = time.time() - t_paper0
        print(f"[DONE PAPER] {arxiv_id} -> accepted_qc_images={per_paper_count} | total={total_images}/{max_images} | {dt:.1f}s", flush=True)

    return images_out_dir, dataset_json_path, counts_csv_path


def main() -> None:
    # ---------------------------------------------------------------------
    # CLI entrypoint
    #
    # This script is the "submission builder":
    # - It processes the given paper list in order.
    # - It downloads/memoizes PDFs into --pdf-dir.
    # - It extracts embedded figures, filters to QC using the QC classifier,
    #   groups figure parts on each page, renders a clean group image, enriches
    #   metadata, predicts gate labels, and stops at --max-images.
    #
    # Outputs are written incrementally to support long runs + resuming:
    # - images_<exam-id>/               (PNG images for submission)
    # - dataset_<exam-id>.json          (final dataset JSON for submission)
    # - paper_list_counts_<exam-id>.csv (counts CSV for submission)
    # ---------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Run the full batch pipeline on a paper list (in order) and build deliverables.")

    # Required "exam identity" and "paper list" inputs.
    ap.add_argument("--exam-id", required=True, help="Your exam id, e.g. 32")
    ap.add_argument("--paper-list", required=True, help="Path to paper_list_<exam_id>.txt")

    # Cache directory for PDFs downloaded from arXiv.
    ap.add_argument("--pdf-dir", default=_default_path("downloaded_pdfs"), help="Cache folder for PDFs")

    # Pretrained QC (quantum-circuit) image classifier checkpoint.
    ap.add_argument("--qc-ckpt", default=_default_path("src/models/best_resnet18_circuit_classifier.pt"), help="QC classifier checkpoint")

    # Pretrained gate patch classifier checkpoint.
    ap.add_argument("--gate-ckpt", default=_default_path("src/models/gate_patch_model.pt"), help="Gate patch model checkpoint")

    # Gate prediction hyperparameters (thresholding + NMS).
    ap.add_argument("--gate-min-conf", type=float, default=0.70)
    ap.add_argument("--gate-iou", type=float, default=0.35)
    ap.add_argument("--gate-pad", type=int, default=6)

    # Rendering of the figure-group bbox from the PDF:
    # higher DPI => higher quality but larger files and slower rendering.
    ap.add_argument("--render-dpi", type=int, default=300, help="DPI for rendering accepted QC bbox regions (lower = smaller/faster)")

    # Resume/fresh behavior:
    # - default: resume from existing dataset/counts files if present
    # - --fresh: ignore existing outputs and start from scratch
    ap.add_argument("--fresh", action="store_true", help="Ignore existing dataset/counts and start from scratch")

    # Optional explicit output paths.
    # If not provided, we create outputs in repo root using exam-id.
    ap.add_argument("--images-out", default="", help="Output folder for PNGs (default: images_<exam-id> in repo root)")
    ap.add_argument("--dataset-json", default="", help="Output dataset json (default: dataset_<exam-id>.json in repo root)")
    ap.add_argument("--counts-csv", default="", help="Output counts CSV (default: paper_list_counts_<exam-id>.csv in repo root)")

    # Batch pipeline knobs.
    ap.add_argument("--max-images", type=int, default=250)
    ap.add_argument("--threshold", type=float, default=0.8, help="QC probability threshold")
    ap.add_argument("--min-size", type=int, default=50, help="Minimum extracted image size")

    # Temporary working directory (raw extracted figures, gate candidate crops, etc.).
    ap.add_argument("--tmp-root", default=_default_path("pipeline_tmp"), help="Temporary working directory")
    args = ap.parse_args()

    # Repo root is used for default output locations (images_<id>, dataset_<id>.json, ...).
    repo_root = _REPO_ROOT
    exam_id = str(args.exam_id)

    # Normalize paths to absolute paths for reproducibility and for running from any CWD.
    paper_list_file = Path(args.paper_list).expanduser().resolve()
    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    qc_ckpt = Path(args.qc_ckpt).expanduser().resolve()
    gate_ckpt = Path(args.gate_ckpt).expanduser().resolve()

    # If user didn't specify explicit output paths, default to:
    #   <repo_root>/images_<exam_id>/
    #   <repo_root>/dataset_<exam_id>.json
    #   <repo_root>/paper_list_counts_<exam_id>.csv
    images_out_dir = Path(args.images_out).expanduser().resolve() if args.images_out else (repo_root / f"images_{exam_id}")
    dataset_json_path = Path(args.dataset_json).expanduser().resolve() if args.dataset_json else (repo_root / f"dataset_{exam_id}.json")
    counts_csv_path = Path(args.counts_csv).expanduser().resolve() if args.counts_csv else (repo_root / f"paper_list_counts_{exam_id}.csv")

    # Keep per-exam temporary artifacts separated:
    # e.g. pipeline_tmp/exam_32/...
    tmp_root = Path(args.tmp_root).expanduser().resolve() / f"exam_{exam_id}"

    # Run the end-to-end batch pipeline.
    # Note: resume=(not args.fresh). With resume enabled, the pipeline:
    # - loads existing dataset JSON and counts CSV if present
    # - skips papers already marked as processed in the counts CSV
    images_out_dir, dataset_json_path, counts_csv_path = run_batch(
        exam_id=exam_id,
        paper_list_file=paper_list_file,
        pdf_dir=pdf_dir,
        qc_ckpt=qc_ckpt,
        gate_ckpt=gate_ckpt,
        images_out_dir=images_out_dir,
        dataset_json_path=dataset_json_path,
        counts_csv_path=counts_csv_path,
        qc_threshold=float(args.threshold),
        min_size=int(args.min_size),
        max_images=int(args.max_images),
        gate_pad=int(args.gate_pad),
        gate_min_conf=float(args.gate_min_conf),
        gate_iou=float(args.gate_iou),
        render_dpi=int(args.render_dpi),
        resume=(not bool(args.fresh)),
        tmp_root=tmp_root,
    )

    print("[DONE]")
    print("images dir :", images_out_dir)
    print("dataset    :", dataset_json_path)
    print("counts csv :", counts_csv_path)


if __name__ == "__main__":
    main()



