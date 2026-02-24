from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import io

# Ensure repo root + src/ are on sys.path (stable imports regardless of CWD)
# File location: <repo_root>/src/pipeline/run_full_pipeline.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for p in (str(_SRC_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    # Package-style imports (preferred)
    from .download_papers import download_papers_batch
    from .extract_figures import extract_figures_from_pdf, ExtractedFigure
    from .enrich_figure_groups import (
        PDFContext,
        group_figures_on_page,
        enrich_figure_group,
        render_bbox,
    )
except Exception:
    # Fallback when executed as a script
    from pipeline.download_papers import download_papers_batch
    from pipeline.extract_figures import extract_figures_from_pdf, ExtractedFigure
    from pipeline.enrich_figure_groups import (
        PDFContext,
        group_figures_on_page,
        enrich_figure_group,
        render_bbox,
    )

# IMPORTANT: QC classifier utilities live in predict/predict_qc_folder.py
from predict.predict_qc_folder import build_model, get_transform, predict_image
from predict.gate_candidate_extractor import save_candidates
from predict.predict_gates_from_qc_image import load_model as load_gate_model, predict_on_crops, nms
from PIL import Image as PILImage

def run(pdf_path: str, ckpt_path: str, out_dir: str, qc_threshold: float = 0.8, min_size: int = 50):
    def _save_png_safe(img: PILImage.Image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img.save(str(path), format="PNG")
            return
        except Exception:
            pass
        try:
            img2 = img.convert("RGB").copy()
            img2.save(str(path), format="PNG")
            return
        except Exception:
            pass
        bio = io.BytesIO()
        img3 = img.convert("RGB").copy()
        img3.save(bio, format="PNG")
        path.write_bytes(bio.getvalue())

    pdf_path = str(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arxiv_id = Path(pdf_path).stem
    raw_dir = out_dir / f"images_raw_{arxiv_id}"
    group_dir = out_dir / f"images_qc_groups_{arxiv_id}"
    raw_dir.mkdir(exist_ok=True)
    group_dir.mkdir(exist_ok=True)

    extracted: List[ExtractedFigure] = extract_figures_from_pdf(
        pdf_path=pdf_path,
        output_dir=str(raw_dir),
        min_size=min_size,
    )

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    img_size = ckpt.get("img_size", 224)

    model = build_model(num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    transform = get_transform(img_size)

    # Gate patch model (CNN)
    gate_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate_ckpt_path = Path(_REPO_ROOT) / "src" / "models" / "gate_patch_model.pt"
    gate_model, gate_idx_to_label, gate_tfm = load_gate_model(gate_ckpt_path, gate_device)

    prob_map: Dict[str, float] = {}
    qc_flags: Dict[str, bool] = {}
    for fig in extracted:
        p = float(predict_image(model, fig.file_path, transform, device))
        prob_map[fig.file_path] = p
        qc_flags[fig.file_path] = (p >= qc_threshold)

    by_page: Dict[int, List[ExtractedFigure]] = {}
    for fig in extracted:
        by_page.setdefault(fig.page_number, []).append(fig)

    dataset: Dict[str, Dict[str, Any]] = {}

    ctx = PDFContext.open(pdf_path)
    try:
        for page_no, figs in sorted(by_page.items()):
            groups = group_figures_on_page(figs)
            for gi, g in enumerate(groups):
                members = g["members"]
                if not any(qc_flags.get(m.file_path, False) for m in members):
                    continue

                page = ctx.doc[page_no - 1]
                # Lower DPI is usually sufficient (no OCR gates) and keeps images smaller.
                img = render_bbox(page, g["bbox"], dpi=300)
                fig_no = rec.get("figure_number")
                fig_str = str(fig_no) if fig_no is not None else "NA"
                group_img_name = f"{arxiv_id}_page{page_no}_fig{fig_str}.png"
                out_path = group_dir / group_img_name
                _save_png_safe(img, out_path)

                rec = enrich_figure_group(
                    fig_group_bbox=g["bbox"],
                    members=members,
                    pdf_path=pdf_path,
                    ctx=ctx,
                )
                rec["_member_probs"] = {Path(m.file_path).name: prob_map.get(m.file_path, None) for m in members}
                rec["_qc_threshold"] = qc_threshold

                # Gate detection: model-only (gate patch CNN)
                cand_dir = out_dir / "_gate_candidates" / Path(group_img_name).stem
                cand_dir.mkdir(parents=True, exist_ok=True)
                cand_json = save_candidates(out_path, cand_dir, pad=6)
                meta = json.loads(Path(cand_json).read_text(encoding="utf-8"))
                candidates = meta.get("candidates", [])
                preds = predict_on_crops(gate_model, gate_idx_to_label, gate_tfm, candidates, device=gate_device)

                strict = {"t": 0.85, "s": 0.85, "z": 0.85, "p": 0.85, "r": 0.85}
                kept = []
                for p2 in preds:
                    lab = p2["pred_label"]
                    thr = strict.get(lab, 0.70)
                    if float(p2["confidence"]) >= thr and lab != "other":
                        kept.append(p2)

                dedup = []
                by_label: Dict[str, List[Dict[str, Any]]] = {}
                for k in kept:
                    by_label.setdefault(k["pred_label"], []).append(k)
                for _, lst in by_label.items():
                    dedup.extend(nms(lst, iou_thr=0.35))

                rec["quantum_gates"] = sorted({d["pred_label"] for d in dedup})
                dataset[group_img_name] = rec
    finally:
        ctx.close()

    dataset_path = out_dir / f"dataset_{arxiv_id}_groups.json"
    dataset_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = out_dir / f"run_report_{arxiv_id}_groups.json"
    report_path.write_text(json.dumps({
        "pdf": str(Path(pdf_path).resolve()),
        "arxiv_id": arxiv_id,
        "qc_groups": len(dataset),
        "qc_threshold": qc_threshold,
        "min_size": min_size,
        "outputs": {
            "group_images_dir": str(group_dir),
            "dataset_json": str(dataset_path),
        },
        "what_changed_v7": {
            "gates_priority": "Try extracting vector text inside figure bbox first; OCR only if no PDF text gates found.",
            "ocr_change": "Removed noisy gate-box detector; uses word OCR only as fallback.",
            "provenance": "quantum_gates are predicted only by the gate patch model (CNN).",
        }
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    return dataset_path, report_path, group_dir

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", help="Path to a local PDF (already downloaded).")
    g.add_argument("--arxiv-id", help="arXiv id to download first (e.g., 2507.23310)")
    ap.add_argument("--pdf-dir", default="downloaded_pdfs")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="pipeline_single_pdf_out")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--min-size", type=int, default=50)
    args = ap.parse_args()

    pdf_path = args.pdf
    if args.arxiv_id:
        pdf_dir = Path(args.pdf_dir); pdf_dir.mkdir(parents=True, exist_ok=True)
        downloaded = download_papers_batch([args.arxiv_id], str(pdf_dir), limit=1)
        if not downloaded:
            raise SystemExit(f"Failed to download arXiv:{args.arxiv_id}")
        pdf_path = downloaded[0]

    ds, rep, gdir = run(pdf_path, args.ckpt, args.out, args.threshold, args.min_size)
    print("\n[DONE]\nDataset:", ds, "\nReport:", rep, "\nGroup images:", gdir)

if __name__ == "__main__":
    main()
