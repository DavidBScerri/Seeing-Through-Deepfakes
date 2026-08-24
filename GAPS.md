# GAPS.md — Honest audit, ordered by severity

Each gap: what it is, where, why it matters, and a fix scoped to a single small task.

> **Prompt 4.5 corrective pass (2026-08-24).** In addition to the historical
> gaps below, a corrective pass addressed a set of semantic drifts introduced
> alongside the C2PA / TrustMark / SHA-256 work:
>
> - Raw C2PA/provenance plumbing (`c2pa`, `claim_generator`,
>   `created_software_agent`, `content credentials`, `manifest`) and
>   ambiguous words (`synthetic`) now contribute exactly zero to the
>   heuristic `P(AI)_m` — previously they nudged the score via
>   `keyword_hits`, `binary_hits`, `software_tag_count`, and
>   `suspicious_only_software_tags`.
> - IPTC digital-source-type mapping is now conservative — only
>   `trainedAlgorithmicMedia` / `compositeSynthetic` are AI-generated,
>   only `compositeWithTrainedAlgorithmicMedia` is AI-modified; the
>   previously-mislabelled `algorithmicMedia`, `dataDrivenMedia`,
>   `algorithmicallyEnhanced`, `screenCapture`, `virtualRecording`,
>   `composite` are treated as UNSPECIFIED.
> - C2PA validation-state semantics fixed: `Trusted` → `valid` with
>   `signer_trusted=True`; `Valid` (and legacy `Untrusted`) →
>   `untrusted_signer`; unknown / `None` → `error`, never `valid`.
>   New `validation_warnings` list separates informational entries
>   from hard failures.
> - "Likely Real" removed from the pipeline output — a fused-negative
>   result is now reported as an inconclusive verdict with `verdict_type="inconclusive"`;
>   historical evaluation reports keep the old label as period terminology.
> - Invalid vs unavailable hash-registry states are preserved end-to-end;
>   `/api/analyse` reports the exact failure with its diagnostic and
>   still includes the uploaded file's SHA-256 digest.
> - TrustMark wrapper now genuinely defaults to CPU and passes
>   `loadRemover=False, loadBBoxDetector=False` (see the caveat that
>   upstream may still initialise components its public API cannot
>   disable).

> **Historical paths.** File locations named below (`src/metadata_module`,
> `src/visual_module`, `src/integration_pipeline`, `src/deepfake_module`)
> are the pre-reshuffle paths as of when each gap was written. The current
> layout is `src/genai_detection/{metadata,visual}_module/`,
> `src/genai_detection/integration_pipeline/`, and `src/deepfake_detection/`
> — see [CLAUDE.md](CLAUDE.md) / [PROJECT.md](PROJECT.md) for the mapping.
> The rest of this file preserves the old names verbatim so the audit
> history and status notes stay legible.

---

## 1. The web app applies a fine-tuned delta to the WRONG base model — SEVERITY: CRITICAL (correctness)

**Status (2026-07-19): FIXED.** `app.py` now derives the base model from the delta checkpoint (`get_delta_base_model`) and loads `run_01_stage2A` (the report's Stage-2 model on `dima806/ai_vs_real_image_detection`, matching bulk_evaluation.ipynb). `load_weight_delta()` raises `ValueError` on base mismatch unless `force=True`. Covered by `tests/test_weight_delta.py`. David confirmed (2026-07-19): `run_01_stage2A` on `dima806/ai_vs_real_image_detection` is canonical everywhere — `visual_classifier.py`/`training.py` defaults and the visual-module notebooks were updated accordingly.

**What:** `src/integration_pipeline/app.py:409–417` loads base `dima806/ai_vs_real_image_detection` and applies `run_02_ft_genimage_w_julienlucas_weight_delta.pt`. That delta's own checkpoint metadata says it was created for `dima806/ai_vs_human_generated_image_detection` (verified: `torch.load(...)["base_model"]`). These are two different HF model repos. `load_weight_delta()` (`src/visual_module/visual_classifier.py:144–149`) detects the mismatch but only **prints a warning** and applies the delta anyway — producing a model that is neither the base nor the fine-tuned one. Every demo verdict and any evaluation run through `app.py` since this pairing is scientifically suspect.

**Why it matters:** This silently corrupts the headline results of the thesis prototype. The deltas exist precisely to reproduce fine-tuned models exactly.

**Fix (single task):** In `app.py`, read `base_model` from the delta checkpoint and pass *that* as `model_name_or_path` (or hard-code the matching base per delta). Additionally change `load_weight_delta()` to `raise ValueError` on base mismatch unless an explicit `force=True` is passed. Confirm with David which pairing his reported run_02 numbers used.

---

## 2. Decision thresholds disagree across every entry point — SEVERITY: HIGH (correctness/reproducibility)

**Status (2026-07-19): FIXED.** `src/integration_pipeline/config.py` centralises the constants, and David confirmed the canonical fused decision threshold is **0.55** (the 0.25 app value was temporary testing from the mispaired-model era). Now 0.55 in config.py, fusion.py default, bulk_evaluation.ipynb, and integration_pipeline.ipynb. The report's printed 0.5 predates this — report 0.55 in the thesis write-up.

**What:** The fused-AI decision threshold is 0.5 in the report (and skill reference), **0.55** default in `fusion.py:109`, **0.25** in `app.py:37` (`wa_threshold`), and **0.45** in `bulk_evaluation.ipynb` config cell. `visual_accuracy` is 0.83 in `fusion.py` vs 0.84 in `app.py`/notebook. `metadata_extraction.py:decide()` uses its own 0.80/0.40 cutoffs.

**Why it matters:** "What counts as AI-generated" changes depending on which file you run. Results across the web app, bulk evaluation, and the written report are not comparable, and a future reader can't tell which value is the documented empirical one.

**Fix (single task):** Create `src/integration_pipeline/config.py` holding one canonical set of constants (weights, accuracies, thresholds) with comments citing the report values; import it from `app.py`, `bulk_evaluation.ipynb`, and use as `fusion.py` defaults. Ask David which threshold is the intended final value before choosing — these are empirical, do not invent.

---

## 3. Zero automated tests — SEVERITY: HIGH

**Status (2026-07-19): FIXED.** `tests/` added (55 tests): fusion strategies pinned to report Annex II (4) values, metadata scoring incl. GAPS #7 regressions, delta round-trip + mismatch guard, multipart parsing, YuNet checksum. `pytest` added to requirements. Run: `python -m pytest tests/`.

**What:** No test files anywhere. The pure-function core — all three `FusionStrategy.fuse()` implementations, `extract_visual_ai_probability()`, `crop_face_region()` (`fusion.py`), `score_features()`, `build_features()`, `flatten_metadata()`, `decide()` (`metadata_extraction.py`), and `save_weight_delta`/`load_weight_delta` round-tripping — is trivially testable without models or network.

**Why it matters:** These functions ARE the thesis contribution. Gap #1 would have been caught by a delta round-trip test; gap #2 by a threshold assertion. Any refactor now is unguarded.

**Fix (single task):** Add `tests/test_fusion.py` and `tests/test_metadata_scoring.py` with pytest: known-input/known-output cases for each fusion strategy (including the crop-max rule and zero-weight edge case), metadata scoring for camera-rich vs AI-claim vs empty metadata (must be ≈0.5 neutral), and a `save_weight_delta`→`load_weight_delta` round-trip on a tiny dummy `nn.Module`. Add `pytest` to requirements.

---

## 4. `requirements.txt` is wrong in both directions — SEVERITY: HIGH (fresh install fails)

**Status (2026-07-19): FIXED.** Added `faiss-cpu`, `matplotlib`, `pandas`, `seaborn`, `pytest`; removed `invisible-watermark`, `PyWavelets`, `scipy`, `birder` (grep-confirmed unused).

**What:** Missing packages that are imported: **`faiss-cpu`** (`deepfake_classifier.py`, `initialise_index.py` — the app cannot start without it), **`matplotlib`** (`training.py`, `gradcam_*`, `app.py` via `_overlay_heatmap`), **`pandas`** (`build_combined_dataset.py`), **`seaborn`** (eval notebook). Listed but never imported anywhere: `invisible-watermark`, `PyWavelets`, `scipy`, `birder` (likely relics of an abandoned watermark-detection direction). ExifTool's external-binary requirement is only mentioned in the README.

**Why it matters:** `pip install -r requirements.txt && python -m src.integration_pipeline.app` fails with `ModuleNotFoundError: faiss` on a clean machine; unused heavy deps slow installs.

**Fix (single task):** Add `faiss-cpu`, `matplotlib`, `pandas`, `seaborn`; remove `invisible-watermark`, `PyWavelets`, `scipy`, `birder` (grep first to confirm still unused); keep the ExifTool note in README.

---

## 5. README describes a module that no longer exists — SEVERITY: MEDIUM (stale docs)

**Status (2026-07-19): FIXED.** README module list matches reality (YuNet, DINOv2+FAISS, occlusion saliency); bulk_evaluation.ipynb intro now documents the `real|ai|deepfake` prefixes and `ai_vs_real_image_detection` base.

**What:** `README.md` says the Deepfake Module includes "scene classification (Places365)". Places365 was removed in commit `28c9503` ("remove scene analysis from deepfake pipeline"). The module is now YuNet + DINOv2/FAISS only. Similarly, `bulk_evaluation.ipynb`'s intro markdown says ground-truth prefixes are `fake`/`human` and the base model is `ai_vs_human_generated_image_detection`, but the code checks `deepfake`/`real`/`ai` prefixes and loads `ai_vs_real_image_detection`.

**Why it matters:** First thing a new reader sees; misleads about actual capabilities and label conventions.

**Fix (single task):** Rewrite the README module list to match reality (YuNet face detection, DINOv2+FAISS landmark retrieval, occlusion saliency) and fix the notebook's intro table to match `infer_ground_truth()` and the actual base-model config.

---

## 6. Hand-rolled multipart/form-data parser — SEVERITY: MEDIUM (fragile edge cases; low security exposure since localhost-only)

**Status (2026-07-19): FIXED.** Multipart parsing now uses the stdlib email parser (binary-safe), uploads capped at 25 MB (413), temp-file suffix derived from the uploaded filename, `numpy` imported at module top, malformed bodies return JSON 400.

**What:** `app.py:71–126` parses multipart bodies with `body.split(boundary)` and regexes.

**Why it matters:** Binary image bytes that happen to contain the boundary sequence corrupt the upload; no `Content-Length` cap means a huge upload is read fully into memory; the temp file is always given a `.png` suffix (`app.py:152`) even for JPEG/WebP, which can skew ExifTool's type-dependent handling and the file-extension-based heuristics. Server binds 127.0.0.1 only, so remote exposure is nil — severity is robustness, not security. (Related, `app.py:169`: `import numpy as np` happens *inside* the first saliency `try:` block but `np` is used again at line 226 in a different block — a failure in the first block leaves `np` undefined in the second.)

**Fix (single task):** Replace the manual parsing with `email.message_from_bytes` or the `multipart` package (or cgi-free equivalent), cap uploads at e.g. 25 MB, derive the temp-file suffix from the uploaded filename, and move `import numpy as np` to the top of the file.

---

## 7. Metadata heuristics have substring false positives — SEVERITY: MEDIUM

**Status (2026-07-19): FIXED.** All AI keywords/binary markers match with word-boundary lookarounds ("influx"/"photosynthetic" no longer hit); `has_camera_claim` is tag-name-based ("Camera Raw" no longer counts); `has_gps` requires non-zero coordinate values (GPSVersionID excluded). Verified against the sample set: only change vs report Annex II (1) is `deepfake-metadata+place-face.png` 0.53→0.50 — its "c2pa" binary hit was random compressed pixel data (ExifTool finds no C2PA), i.e. a confirmed false positive; the decision stays "uncertain".

**What:** In `metadata_extraction.py`: `"flux" in haystack` matches "influx"/"Fluxus"; `has_camera_claim = "captured" in values or "camera" in values` matches "Camera Raw" (an Adobe editor); `"synthetic"` matches "photosynthetic"; `has_gps` (`:203`) is true if any *key* contains "gps" even when GPS values are empty/zeroed; `scan_binary_markers()` greps the entire file body, so any ASCII occurrence of e.g. `flux` in compressed data scores as an AI marker.

**Why it matters:** The metadata stream feeds fusion with `w_m·a_m` weight; systematic false positives on edited-but-real photos push fused scores up. The report already notes metadata "scores 0.99 whenever any AI marker is present" (see `ConservativeThresholdFusion` docstring) — this is part of why.

**Fix (single task):** Use word-boundary regexes for short/ambiguous keywords (`\bflux\b`), require non-empty GPS *values*, restrict `has_camera_claim` to specific tag names rather than substring-of-any-value, and add the false-positive cases to the metadata tests from gap #3.

---

## 8. `gradcam_landmark_analysis.py` uses a non-package import — SEVERITY: MEDIUM (breaks as a module)

**Status (2026-07-19): FIXED.** Package-style import with the same sys.path bootstrap app.py uses; works both as `src.deepfake_module.gradcam_landmark_analysis` and standalone/%run.

**What:** `src/deepfake_module/gradcam_landmark_analysis.py:25` does `from deepfake_classifier import LandmarkIndex` (bare module name), so `from src.deepfake_module.gradcam_landmark_analysis import ...` fails with `ModuleNotFoundError` unless the CWD is the module directory. Its sibling `gradcam_face_analysis.py` avoids this only because it imports nothing local, and `app.py` imports it package-style.

**Why it matters:** Inconsistent import styles mean code works in one notebook and breaks in another; anyone wiring landmark saliency into `app.py` (mirroring the face saliency) hits an immediate import error.

**Fix (single task):** Change to `from src.deepfake_module.deepfake_classifier import LandmarkIndex` with the same `sys.path` bootstrap `app.py` uses (or a relative `.deepfake_classifier` import), and verify both `%run` and package-import paths still work.

---

## 9. Dead code and duplicated logic — SEVERITY: LOW-MEDIUM (tech debt)

**Status (2026-07-19): PARTIALLY DONE.** Vestigial gating params removed from `DeepfakeClassifier.predict()` (all callers pass only the image); app.py's two startup paths share `create_server()`; config constants centralised in config.py. Still open: `crop_face_region`/`_overlay_heatmap` relocation to a shared module, `_get_yunet_model_path` duplication in gradcam_face_analysis.py.

**What:**
- `DeepfakeClassifier.predict(visual_classifier=None, threshold=0.5)` (`deepfake_classifier.py:122–164`): when called without a classifier (as `app.py` does), `ai_score = threshold` so the gate is always true — vestigial gating logic that predates fusion-side gating.
- `app.py` has two server-startup paths: `start_server_thread()` (notebook use) and a near-identical `__main__` block; the config constants at `app.py:32–48` duplicate the defaults in `fusion.py` and the notebook config cells.
- `crop_face_region()` (an image utility) lives in `fusion.py` (decision-math module); `bulk_evaluation.ipynb` imports it mid-loop.
- `_overlay_heatmap()` is duplicated verbatim in `gradcam_face_analysis.py:41` and `gradcam_landmark_analysis.py:34`; `_get_yunet_model_path()` is duplicated in `deepfake_classifier.py:11` and `gradcam_face_analysis.py:374` (the latter without the download).
- The `"AI-generated"` key lookup in `extract_visual_ai_probability()` (`fusion.py:67`) likely never matches either dima806 model's labels, making the docstring's "preferred" path dead and the string-sniffing fallback load-bearing.

**Why it matters:** Each duplicate is a place for constants/behaviour to drift (gap #2 is this disease in action).

**Fix (single task):** Remove the vestigial gating params from `DeepfakeClassifier.predict()`; move `crop_face_region` and `_overlay_heatmap` into a shared `src/common/` (or `visual utils`) module imported everywhere; collapse `app.py`'s two startup paths into one function.

---

## 10. Landmark "confidence" is an uncalibrated cosine similarity treated as a probability — SEVERITY: LOW-MEDIUM (methodological fragility)

**What:** `LandmarkIndex.search()` (`deepfake_classifier.py:197–240`) averages cosine similarities of the top-k hits per label and returns the winner's mean as `confidence`; `predict()` then compares it to the documented `P_lm ≥ 0.5` deepfake threshold. DINOv2 cosine similarities between arbitrary natural images are frequently >0.5, and the index (Google Landmarks v2-mini) has no "not a landmark" background class — an unrelated outdoor photo can clear the bar.

**Why it matters:** `is_deepfake = has_face OR has_place`, so a spurious landmark match alone upgrades "AI-generated" to "Potential Deepfake" — a regulatory-salient distinction in the thesis.

**Fix (single task):** Calibrate: run `LandmarkIndex.search()` over a held-out set of non-landmark images, plot the similarity distribution, and pick/justify a threshold (or add a margin rule: best-label mean minus second-best mean). Report the change in the thesis rather than silently retuning — and confirm the value with David.

---

## 11. Committed large binaries and an 88 GB data directory inside iCloud Drive — SEVERITY: LOW-MEDIUM (operational)

**What:** Git tracks two 84 MB weight deltas, a 60 MB FAISS index, and a 51 MB PowerPoint (near GitHub's 100 MB hard limit; clone size ~350 MB). The gitignored `data/visual/` is 88 GB living in iCloud-synced storage; the Arrow caches (`cache-*.arrow` in `data/visual/julienlucas/`) are regenerable junk that iCloud will happily upload.

**Why it matters:** One more full-model-sized artefact makes the repo unpushable to GitHub; iCloud syncing 88 GB of Arrow shards wastes quota/bandwidth and can corrupt datasets mid-write.

**Fix (single task):** Enable Git LFS for `*.pt`, `*.faiss`, `*.onnx`, `*.ppsx` (or document a HuggingFace Hub upload as the artefact store), and mark `data/visual/` as excluded from iCloud (`.nosync` folder rename or move outside iCloud) — check with David before moving anything.

---

## 12. YuNet ONNX auto-download without checksum — SEVERITY: LOW (supply chain)

**Status (2026-07-19): FIXED.** Download URL pinned to opencv_zoo commit `f12e127…` with SHA-256 verification (hash matches the committed file); existing files are verified on every load and a mismatch warns loudly (never auto-deleted, per the no-overwrite rule).

**What:** `_get_yunet_model_path()` (`deepfake_classifier.py:11–27`) downloads the model from the `opencv_zoo` GitHub `main` branch at runtime if missing, with no hash verification and no pinned revision.

**Why it matters:** A moved/changed upstream file silently changes face-detection behaviour between runs; results stop being reproducible. (The file is currently committed, so the download path is a rarely-exercised fallback — which also means it's untested.)

**Fix (single task):** Pin the URL to a commit SHA and verify a hard-coded SHA-256 after download, raising on mismatch.

---

## 13. Half-finished / abandoned threads — SEVERITY: LOW (inventory, so nobody rediscovers them as "features")

- **Watermark detection was planned, never built:** `invisible-watermark` + `PyWavelets` in requirements with zero imports. Watermarking is one of the CoP transparency layers, so it may return in thesis phase — but today it's dead weight (see gap #4).
- **Fusion strategies 2 and 3 are experiments, not wired to any UI choice:** only editing `app.py`'s `fusion_strategy` constant switches them; no run in `outputs/` exercises them. Fine to keep, but they're untested code paths.
- **`run_03`/`run_04` deltas** exist in `fine_tuned_model_delta/` with no corresponding eval JSON in `src/visual_module/outputs/` — undocumented experiments; ask David whether they're keepers before building on them.
- **Old CSV references stale filenames:** `outputs/bulk_evaluation_results_run_02_*.csv` uses `fake+…` filenames; samples were later renamed to `deepfake+…` (commit `a0a44d3`). Re-running the notebook on today's folder won't reproduce that CSV row-for-row.
- **`data/.DS_Store`, `src/.DS_Store` etc.** exist on disk (correctly gitignored, just noise).

**Fix (single task each):** documented above; the cheapest is a one-line note in each stale CSV's name or a `outputs/README.md` mapping runs → deltas → sample-set versions.

---

## Explicitly checked and NOT gaps

- **Secrets:** none committed. HF token is read from `HF_TOKEN` env or interactive login only. `.claude/settings.local.json` contains only permission rules.
- **Server exposure:** binds `127.0.0.1` only; appropriate for a demo tool.
- **Missing-metadata handling:** correctly scores ~0.5 neutral (design requirement), not "real".
- **`WeightedAverageFusion` math** faithfully implements the report's Stage 0–1 formulas (including the crop-max rule and w/a separation); the divergences are the *threshold constants* (gap #2), not the formula.
