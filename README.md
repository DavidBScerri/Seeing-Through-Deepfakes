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
- **Metadata Module** — Two independent evidence streams:
  1. **Heuristic EXIF/binary indicators** (`metadata_extraction.py`) — ExifTool-driven scan for camera fields, AI provider keywords (openai, midjourney, …), suspicious software-only tags, and binary markers in the file bytes. Feeds `P(AI)_m` into fusion.
  2. **Cryptographic C2PA provenance** (`provenance_validation.py`) — uses the official `c2pa-python` library to actually validate embedded manifests. Produces a typed `ProvenanceResult` with a `ProvenanceStatus` (`valid`, `invalid_or_tampered`, `untrusted_signer`, `absent`, `unsupported_format`, `validator_unavailable`, `error`) and an `OriginClaim` (`ai_generated`, `ai_modified`, `camera_or_human_origin`, `unspecified`, `conflicting`). Reported as a separate evidence object — deliberately NOT folded into the fusion formula for this iteration.
- **Deepfake Module** — Face detection (YuNet), landmark retrieval (DINOv2 + FAISS), and occlusion-based saliency maps for explainability.
- **Integration Pipeline** — Decision-fusion engine with a web interface that combines all module outputs into a single verdict.

### What "provenance" actually means here

The metadata module now cleanly separates five layers so nothing gets conflated:

| Layer | Meaning | Trust |
|---|---|---|
| 1. Raw EXIF/XMP | Camera and software fields (Make, Model, CreatorTool, GPS, …) | Freely writable — indicator only |
| 2. C2PA marker | The strings `c2pa`, `claim_generator`, `created_software_agent` appear somewhere in the file | Indicator only — camera-signed and AI-generated images both carry these |
| 3. Cryptographically valid C2PA manifest | The `c2pa-python` validator opened the manifest and its signatures verified | Trust rests on the signer chain |
| 4. Explicit AI-generation / AI-modification assertion | A valid manifest whose `c2pa.actions` carry an IPTC `trainedAlgorithmicMedia` / `algorithmicallyEnhanced` / … `digitalSourceType` | The signer *asserts* AI involvement |
| 5. Substantive truth | Whether the image really is AI-generated | **Not something this framework can guarantee.** Even a validated manifest is only as trustworthy as its signer, and a signer can lie. |

Missing metadata is always **inconclusive** — never "real". A missing validator is reported as `validator_unavailable`, never as `absent` — the two are distinct statuses so a broken toolchain cannot be mistaken for "no manifest found".

## Setup

```bash
pip install -r requirements.txt
```

ExifTool must be installed separately for the metadata module (`brew install exiftool` on macOS). The C2PA validator (`c2pa-python`) is pulled in by `requirements.txt`; when it is missing the provenance panel reports `validator_unavailable` rather than falling back to a heuristic guess.

## Tests

```bash
python -m pytest tests/
```

## Running

Launch the web interface:

```bash
python -m src.genai_detection.integration_pipeline.app
```
