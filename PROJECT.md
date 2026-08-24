# PROJECT.md — Seeing Through Deepfakes

## What this is

A proof-of-concept **AI-image / deepfake detection pipeline** built by David Scerri for his M.Sc. dissertation *"Seeing Through Deepfakes"* (developed during a work placement with the Malta Digital Innovation Authority, extended for the thesis). It takes a single image and produces a **decision-support verdict** — `Inconclusive` (fused probability below the decision threshold — a negative result is NOT proof of camera or human origin), `Likely AI Generated`, or `Potential Deepfake` — with a probability and a human-readable explanation for every contributing signal. Historical evaluation runs (Annex II of the placement report) used the earlier `Likely Real` label — retained in those historical results only as period terminology, no longer emitted by the current pipeline.

Crucially, this is not "just a classifier." The thesis argument is regulatory: the EU AI Act **Article 50(2)** requires generative AI outputs to be detectable through machine-readable means. Layered transparency (provenance metadata, watermarking, model-based detection) was previously discussed under the Draft Code of Practice on Transparency of AI-Generated Content — the current legal / regulatory instruments (including the version and status of that Code) evolve, and any reference to a specific draft belongs in the thesis's discussion section rather than as a live description of settled law. The pipeline mirrors that layering pragmatically: several imperfect evidence streams are combined — the two that currently feed **fusion** (`P(AI)_m`, `P(AI)_v*`) and separate structured evidence objects (C2PA provenance, TrustMark watermark, SHA-256 registry) that report standalone — none is authoritative, and every module exposes a probability + rationale rather than a binary verdict. The Art. 3(60) deepfake definition covers *persons, objects, places, entities or events* — which is why the deepfake stage checks for both faces **and** landmark/place resemblance, not just faces.

The audience is David (sole developer), thesis examiners, and demo viewers. It is research code: single-user, localhost-only, no deployment story — but the *formulas and thresholds are the scientific contribution*, so correctness of the fusion math matters more than typical prototype code.

## Tech stack and why

| Piece | Role | Why chosen |
|---|---|---|
| Python 3.10+ (uses `X \| None` unions) | Everything | Standard ML research stack |
| PyTorch + HuggingFace `transformers` | ViT visual classifier, DINOv2 embeddings | Reproducible pretrained models; `dima806/*` base gave a documented starting point |
| HuggingFace `datasets` (Arrow on disk) | Training/eval data | Streaming + `save_to_disk` for the ~88 GB `data/visual/` corpora |
| OpenCV `FaceDetectorYN` (**YuNet** ONNX) | Face detection | Replaced MTCNN (commit `b4b11ef`); lightweight, no torch dependency, per-image input sizing |
| **DINOv2-base + FAISS** (`IndexFlatIP` over L2-normalised CLS embeddings ⇒ cosine similarity) | Landmark/place retrieval | Semantic "does this resemble a known real place" per Art. 3(60) |
| **ExifTool** (subprocess, must be installed separately) | Metadata extraction | The de-facto standard for EXIF/XMP/C2PA dumping |
| Pydantic v2 | `FeatureSet` / `AnalysisResult` models | Typed, serialisable module outputs |
| `http.server` stdlib + one 1,900-line `index.html` | Demo web UI | Deliberately zero web-framework dependencies; single-file dark-themed SPA |
| Jupyter notebooks | Training, evaluation, demos | The interactive "workbench" pattern — notebooks orchestrate, `.py` files hold reusable logic |

Tests live under `tests/` and run with `python -m pytest tests/` (208 tests as of this pass, covering fusion strategies, metadata scoring, C2PA provenance validation, TrustMark, SHA-256 hash registry, weight-delta round-tripping, website plumbing and package layout). No linter config, no CI, no packaging (no `pyproject.toml`) yet.

## Architecture

```
                         Input image
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
 Metadata & Provenance Module              Visual Classifier (ViT)
 src/genai_detection/metadata_module/      src/genai_detection/visual_module/
 exiftool → feature flags →               base model + int8 weight-delta
 heuristic score  P(AI)_m                  → P(AI)_v
        │                                         │
        │            YuNet face detected?  → crop → reclassify → P(AI)_vcf
        │                                         │
        └──────────────┬──────────────────────────┘
                       ▼
              Decision Fusion  (src/genai_detection/integration_pipeline/fusion.py)
              P(AI)_v* = max(P(AI)_v, P(AI)_vcf)
              weighted / conservative / bayesian strategy → P(AI), is_ai
                       │ is_ai == True  (proportionality gate)
                       ▼
              Conditional Deepfake Module (src/deepfake_detection/)
              YuNet face (P_f ≥ 0.75?)  +  DINOv2+FAISS landmark (P_lm ≥ 0.5?)
              is_deepfake = has_face OR has_place
                       │
                       ▼
     Verdict: "Inconclusive" | "Likely AI Generated" | "Potential Deepfake"
             + per-module probabilities, rationale, saliency heatmaps
```

### The four modules

1. **`src/genai_detection/metadata_module/`** — two independent evidence streams share the module:
   - **`metadata_extraction.py`** runs `exiftool -j -a -u -g1`, flattens the JSON to lowercase `key → value` strings, scans the raw file bytes for binary markers (`c2pa`, `openai`, `midjourney`, …), and derives boolean features (`has_make`, `has_c2pa_marker`, `has_ai_claim`, `suspicious_only_software_tags`, …). `score_features()` starts at **0.50 (neutral)** and adds/subtracts per feature (+0.35 explicit AI claim, −0.20 rich camera metadata, clamp to [0.01, 0.99]). This encodes the core design rule: **absence of metadata is uncertainty, not authenticity** — platforms strip metadata routinely. `decide()` maps to `likely_ai_generated` / `likely_camera_origin` / `uncertain`. Pure heuristics, no ML, fully explainable by construction. Critically, `has_ai_claim` is NOT set by `c2pa`, `claim_generator`, or `created_software_agent` — those tag names identify the provenance system, not the origin of the content.
   - **`provenance_validation.py`** cryptographically validates embedded C2PA manifests via the official `c2pa-python` library and returns a typed `ProvenanceResult`. Missing manifest → `absent`; library missing → `validator_unavailable` (never confused with absent); manifest present but validation failed → `invalid_or_tampered`; valid manifest with an IPTC `trainedAlgorithmicMedia`-family `digitalSourceType` on a `c2pa.created` action → `OriginClaim.AI_GENERATED`. **Provenance is exposed as a separate evidence object; it is deliberately NOT folded into `P(AI)_m` in this iteration** because doing so would change the reported fusion formula without David's sign-off. See `models.py` for the enum + result schema.

2. **`src/genai_detection/visual_module/`** — `VisualClassifier` wraps a HuggingFace ViT image classifier. Fine-tuned weights are stored as **int8-quantised weight deltas** (`save_weight_delta`/`load_weight_delta` in `visual_classifier.py`): only the diff vs. the base model is committed (~27–84 MB instead of ~330 MB), and it's applied in-place at load. `training.py` holds the Trainer helpers (layer freezing, experience replay, augmentation, early stopping); the actual training runs live in `visual_classifier_finetuning.ipynb` (a config-cell-driven "workbench"). `build_combined_dataset.py` assembles the training corpus from three HF sources (JulienLucas, NTIRE, DeepfakeJudge), balanced 50/50 real:AI, split 80/10/10 — **keep this methodology when retraining or results stop being comparable** to the report's Figures 2–3 (base model: acc 0.523, recall 0.064 → fine-tuned: acc 0.841, recall 0.846 on the combined test split).

3. **`src/deepfake_detection/`** — `DeepfakeClassifier` = YuNet face detector (auto-downloads the ONNX on first use) + `LandmarkIndex` (DINOv2 CLS embeddings searched against a FAISS index of Google Landmarks v2-mini, built by `initialise_index.py`). It does **not** re-decide real vs. AI; it adds semantic context (face-like? place-like?) to images already flagged. The two `gradcam_*_analysis.py` files are misnomers — they implement **occlusion-based saliency** (slide a mean-colour patch over the image, measure confidence drop), used for the report's explainability figures and the web UI heatmaps.

4. **`src/genai_detection/integration_pipeline/`** — `fusion.py` defines three `FusionStrategy` classes: `WeightedAverageFusion` (the thesis formula: reliability-weighted average with separate importance weights `w` and accuracy scalars `a`, normalised), `ConservativeThresholdFusion` (AND-gate), `BayesianFusion` (independent-evidence Bayes). `app.py` is the demo backend: stdlib HTTP server, hand-rolled multipart parsing, serves `web/index.html` (the frontend now lives outside `src/`), exposes one endpoint `POST /api/analyse`, and reads its live configuration constants (weights, thresholds, strategy choice) from `config.py`. `bulk_evaluation.ipynb` runs the whole pipeline over `data/sample_images/` and exports CSVs to `outputs/`.

### The exact fusion math (the load-bearing part)

From the work placement report §3.4 — do not approximate:

- Stage 0: `P(AI)_v* = max(P(AI)_v, P(AI)_vcf)` — a strong face-crop signal must not be diluted by background.
- Stage 1: `P(AI) = (w_m·a_m·P(AI)_m + w_v·a_v·P(AI)_v*) / (w_m·a_m + w_v·a_v)` — `w` = importance, `a` = reliability; a new signal gets its own `w`/`a` pair in both numerator and denominator.
- Stage 2 (documented): `P(AI) ≥ 0.5 → D₂` (AI-generated) else `D₁` (not AI). Only `D₂` triggers Module 4.
- Stage 3: `D₃` (probable deepfake) iff `P_f ≥ 0.75 ∨ P_lm ≥ 0.5` — OR, because Art. 3(60) covers places as well as persons.

**Known, documented failure mode** (report it, don't silently fix): the weighted average is dominated by a confidently-wrong visual classifier — deepfake samples with `P(AI)_v ≈ 0.01` get pulled below threshold even when metadata says 0.99. `ConservativeThresholdFusion` and `BayesianFusion` exist as experimental responses. Any robustness fix must stay an interpretable evolution of this mechanism (e.g. disagreement-aware `a_v`), not a black-box learned combiner.

### Data flow at inference (web app)

`index.html` uploads a file → `app.py:run_analysis_pipeline()` → temp file for exiftool → metadata score → full-image ViT prediction → whole-image YuNet occlusion saliency (base64 heatmap for UI) → YuNet face detect → crop (30% padding) → ViT on crop + crop saliency → fusion → if `is_ai`: `DeepfakeClassifier.predict()` → JSON response with verdict + everything above.

## Key design decisions (inferred and documented)

- **Fusion over any single detector.** The whole system architecture is the thesis claim. Never add a signal that bypasses fusion.
- **Neutral-on-missing metadata** (score starts at 0.5) — regulatory framing: a false "real" from stripped metadata is the worst error.
- **Proportionality gating** — the expensive semantic stage only runs on fused-positive images. Keep new expensive stages behind similar gates.
- **Weight deltas in git instead of full models** — reproducible fine-tuned models in a normal-sized repo; the delta records its base model name and load warns (only warns!) on mismatch.
- **Notebooks orchestrate, modules implement** — every module has an `_eval.ipynb`; shared logic was progressively extracted to `.py` files (see commits `33aba45`, `67c8b66`).
- **Explainability everywhere** — rationale lists, `explanation` dicts on every `FusionResult`, occlusion saliency maps. This is Article-50-driven, not cosmetic.
- **British English naming** — `analyse_image`, `initialise_index.py` (renamed for consistency in `da55917`), "Visualisation".

## Critical paths (ranked)

1. **`src/genai_detection/integration_pipeline/fusion.py`** — the scientific contribution. Changing formulas/thresholds silently changes what counts as a deepfake and invalidates report figures. Highest care.
2. **`src/genai_detection/metadata_module/metadata_extraction.py` `score_features()`/`decide()`** — the metadata evidence stream; same reasoning.
3. **`src/genai_detection/visual_module/visual_classifier.py` delta save/load** — a wrong base-model pairing corrupts every downstream number (this bug currently exists in `app.py` — see GAPS.md #1).
4. **`app.py` configuration block (lines ~32–48)** — the *live* thresholds and weights for the demo. These currently disagree with the documented formula values.
5. **`fine_tuned_model_delta/*.pt` and `src/deepfake_detection/models/*`** — committed experiment artefacts; regenerating them exactly is expensive or impossible. Never delete or overwrite.
6. Safe to change casually: `index.html` styling, saliency visualisation cosmetics, notebook display cells, print formatting.

## Surprises / things that will trip you up

- **The deployed canonical visual backbone is `OwensLab/commfor-model-384`** (David, 2026-08-12). The fine-tuned ViT on `dima806/ai_vs_real_image_detection` paired with `run_01_stage2A` is retained as a HISTORICAL / comparison model — used by `visual_classifier.py`/`training.py` defaults, the visual-module notebooks, and referenced by `run_01_stage2A` (the report's Stage-2 model). The run_02–04 deltas were trained on the *other* repo (`ai_vs_human_generated_image_detection`) and only load on that base; `load_weight_delta` raises on mismatch and `get_delta_base_model` reads the recorded base.
- **"gradcam" files contain no Grad-CAM** — occlusion saliency replaced Grad-CAM in `b4b11ef`; filenames were kept.
- **The canonical fused decision threshold is 0.55** (David, 2026-07-19; centralised in `src/genai_detection/integration_pipeline/config.py`). The report's printed 0.5 predates this decision; the old 0.25 (app) / 0.45 (bulk notebook) were testing values from the mispaired-model era.
- **`DeepfakeClassifier.predict()`'s internal visual gating is a no-op** as called from `app.py`: without a `visual_classifier` arg, `ai_score = threshold`, so the gate always passes. The real gate is fusion's `is_ai` in the caller.
- **Sample-image filename convention** encodes ground truth: `{real|ai|deepfake}{+|-}metadata{+|-}place{+|-}face.ext` (`+` = has that property). `bulk_evaluation.ipynb` infers labels from the prefix.
- **The repo lives inside iCloud Drive** with an 88 GB gitignored `data/` directory. Paths contain spaces (`Mobile Documents`, `com~apple~CloudDocs`) — always quote them in shell commands.
- **ExifTool is an external binary** (`brew install exiftool`); without it the metadata module degrades to a neutral 0.5 score with `metadata_extracted: false`.
- **The FAISS "confidence" is an average cosine similarity** over top-k hits for the winning label, not a calibrated probability — yet it's compared against the 0.5 `P_lm` threshold.
- YuNet detects **face-like structure**, not identity — the report notes a LEGO figure detected as a face. Treat `has_face` as "face-like content present", never "depicts a real person".
- A companion **Claude skill** (`deepfake-detection-framework`) exists outside this repo with the report's exact formulas transcribed (`references/architecture.md`, `references/fusion-formula.md`, `references/report-style.md`) and a report-style chart script (`scripts/eval_report.py`). Thesis PDFs are in `docs/`.
