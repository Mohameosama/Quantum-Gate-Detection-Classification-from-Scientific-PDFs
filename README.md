# Quantum Gate Detection & Classification from Scientific PDFs

A computer vision pipeline for **detecting** and **classifying** quantum gates directly from **real quantum circuit figures** extracted from arXiv PDFs.

This project is motivated by a practical gap: quantum computing papers contain a huge number of circuit diagrams, but there is no widely used, public dataset for **gate detection**, **quantum circuit OCR**, or **structured circuit extraction** from real-world figures. Most prior work uses simulated or ideal diagrams.

This repo addresses the gap by:
- Extracting circuit figures from scientific PDFs (arXiv `quant-ph`)
- Proposing gate candidates using classical CV (OpenCV)
- Building a manually labeled gate-patch dataset
- Training a CNN classifier for gate recognition
- Running gate prediction on newly extracted circuit figures

## Pipeline (end-to-end)

```
PDF
  -> circuit figure extraction
  -> circuit filtering (QC vs non-QC)
  -> gate candidate detection (classic CV)
  -> gate patch cropping
  -> manual labeling
  -> CNN training
  -> gate prediction (per figure)
```

## Repository layout

```text
nlp_project/
  src/
    pipeline/               # End-to-end run: download -> extract -> filter -> enrich -> export
    predict/                # Gate candidate extractor + gate patch inference
    train/                  # Gate patch classifier trainer (+ QC classifier trainer)
    data_preparation/       # Patch extraction + label file builders
    models/                 # Checkpoints used for inference

  # Import shims (allow python -m pipeline..., python -m predict..., etc.)
  pipeline/
  predict/
  train/
  data_preparation/

  requirements.txt

  # Example outputs already included in this repo
  images_32/                # Example exported circuit figures
  dataset_32.json           # Example dataset JSON (metadata + predicted gates)
  paper_list_32.txt
  paper_list_counts_32.csv
```

## Setup

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If `torch` / `torchvision` installation fails, follow the guidance inside `requirements.txt` for CPU vs CUDA wheels.
- `pytesseract` is only used by some scripts under `src/archive/`. If you run those, you also need the system `tesseract` binary.

## Usage

### 1) Build a circuit-figure dataset from arXiv PDFs (batch)

The batch entrypoint is `src/pipeline/run_full_pipeline_batch.py` (CLI via shim: `python -m pipeline.run_full_pipeline_batch`).

Example (the repo already contains inputs for run id `32`):

```bash
python -m pipeline.run_full_pipeline_batch \
  --exam-id 32 \
  --paper-list paper_list_32.txt \
  --threshold 0.8 \
  --max-images 250
```

Outputs:
- `images_32/`
- `dataset_32.json`
- `paper_list_counts_32.csv`
- temp artifacts under `pipeline_tmp/exam_32/`

Resume is enabled by default. For a fresh run:

```bash
python -m pipeline.run_full_pipeline_batch \
  --exam-id 32 \
  --paper-list paper_list_32.txt \
  --fresh
```

### 2) Process a single PDF (debug/development)

```bash
python -m pipeline.run_full_pipeline \
  --pdf downloaded_pdfs/2403.12112.pdf \
  --ckpt src/models/best_resnet18_circuit_classifier.pt \
  --out pipeline_single_pdf_out \
  --threshold 0.8
```

### 3) Gate candidate detection (classic CV)

Implementation:
- `src/predict/gate_candidate_extractor.py`

Key detectors:
- `detect_gate_boxes()`
- `detect_control_dots()`
- `detect_target_plus()`

Batch extraction (produce crops + a merged index):

```bash
python -m data_preparation.batch_extract_gate_patches \
  --qc-root images_32 \
  --out-root pipeline_tmp/gate_patches_out \
  --pad 6
```

### 4) Manual labeling and `labels.json`

This project assumes a manual labeling step for high-quality supervision:
- Create a folder structure like:

```text
my_labeled_patches/
  h/
  x/
  z/
  rx/
  ry/
  rz/
  s/
  t/
  control_dot/
  target_plus/
  other/
```

Then build `labels.json`:

```bash
python -m data_preparation.build_labels_from_folders \
  --labeled-root my_labeled_patches \
  --out labels.json
```

### 5) Train the gate patch classifier (CNN)

Trainer:
- `src/train/train_gate_patch_classifier.py`

```bash
python -m train.train_gate_patch_classifier \
  --labels labels.json \
  --out src/models/gate_patch_model.pt \
  --epochs 25 \
  --img 128
```

### 6) Run gate prediction on a circuit image

```bash
python -m predict.predict_gates_from_qc_image \
  --image images_32/<some_image>.png \
  --model src/models/gate_patch_model.pt \
  --out pipeline_tmp/prediction.json
```

Or on a whole folder:

```bash
python -m predict.batch_predict_gates_from_folder \
  --images-dir images_32 \
  --model src/models/gate_patch_model.pt \
  --out-dir pipeline_tmp/gate_preds_32 \
  --min-conf 0.70 \
  --iou 0.35
```

## Dataset (gate patch labels)

The gate patch dataset is created from real circuit figures extracted from arXiv PDFs. There is no ready-to-use dataset (in this format) for gate detection and classification from noisy, real-world paper figures, so patches are collected and labeled from scratch https://www.kaggle.com/datasets/mosamab/labeled-quantum-gates/data.

### Classes (example distribution)

| Label | Count |
|---|---:|
| control_dot | 257 |
| h | 187 |
| other | 1027 |
| rx | 101 |
| ry | 105 |
| rz | 105 (includes P/phase gates) |
| s | 33 |
| t | 63 |
| target_plus | 135 |
| x | 108 |
| z | 35 |

Notes:
- `control_dot` is the control qubit indicator.
- `target_plus` is the CNOT target symbol.
- `rz` includes P/phase gates.
- `other` contains ambiguous or rare gate-like shapes.

## Challenges

- Very small gate sizes
- Visual similarity between gate symbols
- Style variation across papers
- Noisy PDF extractions
- Severe class imbalance
- Time-intensive manual labeling

## Roadmap

- Expand dataset coverage (more papers, more styles)
- Add missing gates: U, U1, U2, U3; measurement; Y; rarer symbols
- Improve candidate filtering to reduce false positives
- Extend from gate tags to full circuit reconstruction (diagram-to-code)
- Publish a cleaned dataset release (e.g., Kaggle/Zenodo) with documentation

## Research applications

- Quantum circuit OCR
- Diagram-to-code conversion
- Scientific PDF understanding
- Multimodal pipelines (LLM + CV) for quantum research

## Contributing

Contributions are welcome:
- New labeled gate patches
- Additional gate classes
- Model improvements and training recipes
- Benchmark comparisons on real extracted figures

If you want, I can also add:
- A clean architecture diagram
- Example prediction images / failure cases
- A more academic-style writeup (methods, ablations, limitations)
