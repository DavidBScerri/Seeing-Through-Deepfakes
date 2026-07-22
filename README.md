# Seeing Through Deepfakes

A multi-module detection system that combines visual classification, metadata forensics, and semantic analysis to identify AI-generated and deepfake imagery.

## Modules

- **Visual Module** — Fine-tuned ViT classifier for distinguishing AI-generated images from real photographs (fine-tuned weights stored as int8 weight deltas).
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
python -m src.integration_pipeline.app
```
