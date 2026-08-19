# Seeing Through Deepfakes

A multi-module detection system that combines visual classification, metadata forensics, and semantic analysis to identify AI-generated and deepfake imagery.

## Project layout

```
src/
├── genai_detection/            # Generative-AI evidence streams
│   ├── metadata_module/        # EXIF/C2PA heuristics (ExifTool)
│   ├── visual_module/          # ViT / Community Forensics classifier
│   ├── integration_pipeline/   # Decision-fusion engine + web backend (app.py)
│   ├── watermark_module/       # (stub — reserved for later)
│   ├── hash_module/            # (stub — reserved for later)
│   └── evaluation/             # (stub — cross-module robustness harness)
└── deepfake_detection/         # Conditional deepfake stage
    ├── deepfake_classifier.py  # YuNet face + DINOv2/FAISS landmark
    ├── gradcam_*_analysis.py   # Occlusion saliency (misnamed for history)
    └── models/                 # YuNet ONNX + FAISS landmark index
web/
└── index.html                  # Frontend served by integration_pipeline/app.py
tests/
```

## Modules

- **Visual Module** — Fine-tuned ViT / Community Forensics classifier for distinguishing AI-generated images from real photographs (fine-tuned weights stored as int8 weight deltas).
- **Metadata Module** — EXIF/C2PA metadata extraction and heuristic scoring via ExifTool.
- **Deepfake Module** — Face detection (YuNet), landmark retrieval (DINOv2 + FAISS), and occlusion-based saliency maps for explainability.
- **Integration Pipeline** — Decision-fusion engine with a web interface that combines all module outputs into a single verdict.

## Setup

```bash
pip install -r requirements.txt
```

ExifTool must be installed separately for the metadata module (`brew install exiftool` on macOS).

## Tests

```bash
python -m pytest tests/
```

## Running

Launch the web interface:

```bash
python -m src.genai_detection.integration_pipeline.app
```
