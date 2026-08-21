# Seeing Through Deepfakes

A multi-module detection system that combines visual classification, metadata forensics, and semantic analysis to identify AI-generated and deepfake imagery.

## Project layout

```
src/
├── genai_detection/            # Generative-AI evidence streams
│   ├── metadata_module/        # EXIF/C2PA heuristics (ExifTool)
│   ├── visual_module/          # ViT / Community Forensics classifier
│   ├── integration_pipeline/   # Decision-fusion engine + web backend (app.py)
│   ├── watermark_module/       # Adobe TrustMark detector (scheme-specific)
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
- **Watermark Module** — Scheme-specific Adobe TrustMark detector wrapping the official [`trustmark`](https://github.com/adobe/trustmark) library (see [Watermark Module (TrustMark)](#watermark-module-trustmark) below).
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

## Watermark Module (TrustMark)

`src/genai_detection/watermark_module/` wraps Adobe's official
[`trustmark`](https://github.com/adobe/trustmark) library and exposes a
typed `TrustMarkResult` (see `models.py`) surfaced by `/api/analyse`
under the `watermark` key and rendered as a **Watermark Evidence** card
in the web UI. It is deliberately scheme-specific — this module speaks
only about Adobe TrustMark watermarks — and it is **not** wired into the
fusion formula in this iteration (adding a `w`/`a` pair requires
separate evaluation and David's sign-off, per the fusion-formula rule
in `CLAUDE.md`).

- **Supported variants:** whatever the installed `trustmark` library
  actually implements — today `Q`, `P`, `B`, `C` (the `model_type`
  values the upstream `TrustMark(...)` constructor accepts). `Q` is the
  default and mirrors the library's own default. Other schemes named in
  the Content Authenticity ecosystem (Google DeepMind SynthID, Meta
  Stable Signature, vendor-proprietary watermarks) are **not** covered
  — asking about them here returns `unsupported`, never `not_detected`.
- **Model download and cache:** the `trustmark` package downloads its
  own model weights (from Adobe's S3 host, with MD5 verification) on
  first use into its own cache directory adjacent to the installed
  package. This repository never commits TrustMark weights, and never
  downloads them during module import. Weight loading is **lazy** — the
  first `TrustMarkDetector.analyse(...)` call for a variant does the
  work; subsequent calls reuse the cached model instance. To relocate
  the cache, point the `trustmark` package at your preferred directory
  through its own configuration.
- **Absence is inconclusive.** A `not_detected` result means only "no
  supported TrustMark watermark was decoded from this image". It does
  NOT mean the image is unwatermarked, real, or free of AI generation
  — every result carries a fixed `scope_statement` making this
  explicit. Detector-unavailable is reported separately from
  not-detected so a broken toolchain (missing library, unreachable
  model host) cannot be mistaken for "no watermark found".
- **Result states:** `detected`, `not_detected`, `unsupported`,
  `detector_unavailable`, `error` — see `models.TrustMarkStatus` for
  their meanings.
- **Optional integration test.** The default suite mocks the
  `trustmark` library so it runs offline. An opt-in end-to-end test
  encodes a payload into a solid-colour cover with the real library,
  decodes it, and asserts an unwatermarked control is not reported as
  detected. Enable with:

  ```bash
  pip install trustmark
  python -m pytest -m integration tests/test_trustmark_detector.py
  ```

  The test skips (rather than fails) if `trustmark` is not installed or
  its weights cannot be downloaded, so CI on a locked-down box stays
  green.

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
