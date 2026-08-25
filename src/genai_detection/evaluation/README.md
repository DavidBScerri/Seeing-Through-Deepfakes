# Robustness experiment — transparency signals under transformation

This directory holds the cross-module robustness harness for the three
public, machine-readable transparency signals this project ships:

- **C2PA / metadata provenance** — cryptographic manifest validation
  ([`provenance_validation.py`](../metadata_module/provenance_validation.py)).
- **Adobe TrustMark** — the only invisible watermark scheme the pipeline
  supports today
  ([`trustmark_detector.py`](../watermark_module/trustmark_detector.py)).
- **SHA-256 byte-exact hash** — text-only registry lookup
  ([`hash_module`](../hash_module/)).

It is a small, reproducible experiment — not a benchmark against every
possible provider. It answers one question honestly for the mechanisms
we implement, and clearly says "unavailable" for anything else.

## Research question

> Under a set of common image transformations (byte-preserving copy,
> metadata stripping, JPEG re-encoding, resizing, centre cropping,
> format conversion, a screenshot-like rasterisation, and a mild
> brightness adjustment), how does each of the three implemented
> transparency signals behave?

The results are read side-by-side, per signal, per transformation. We
never combine them into a single "accuracy" number — SHA-256's byte-
exact answer, C2PA's cryptographic status, and TrustMark's watermark
survival answer different questions.

## What this evaluates — and what it does not

Evaluates:

- The **implemented** validators and detectors this project ships.
- Their behaviour on transformations you can reasonably encounter in
  the wild (a JPEG re-save, a resize, a metadata strip).

Does **not** evaluate:

- SynthID, Meta Stable Signature, or any other invisible-watermark
  scheme. This repository has no detector for them; inventing a
  "not-found" status would be dishonest.
- Whether an image is AI-generated. A TrustMark, a hash registration,
  or a C2PA claim are transparency signals, not truth. In particular:
  a TrustMark embedded at experiment time only proves the experiment
  embedded a TrustMark. Content origin is out of scope for this
  harness.
- Fusion, thresholds, weights. The runner does not read `fusion.py`
  and cannot change any decision made downstream.

## Dataset and sample selection

Default source: the small `data/sample_images/` set already used by
`bulk_evaluation.ipynb`. Ground truth is either implicit in the
filename convention (`{real|ai|deepfake}{+|-}metadata...`) or derived at
runtime — the harness runs C2PA validation on each source ONCE and uses
the result as the "baseline" against which every transformation is
compared.

You can point `--input-dir` at your own directory (subject to
licensing; the repo commits only its own tiny sample set).

The runner never copies your source images into `outputs/`, `data/` or
git; every derivative lives in a `tempfile.TemporaryDirectory` that is
removed on exit, both on success and on failure.

## Transformations

All parameters live in [`transformations.py`](transformations.py). The
runner records the exact spec in `experiment_config.json`.

| Name                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `original_copy`         | Byte-preserving copy of the source (control).                               |
| `metadata_stripped`     | Re-encode in source format with no EXIF/XMP block.                          |
| `jpeg_q90` / `q70` / `q50` | JPEG re-encode at the named quality.                                     |
| `resize_75` / `resize_50` | Lanczos resize to 75% / 50% of each dimension (JPEG-95 output).           |
| `crop_5pct` / `crop_10pct` | Centre crop removing ~5% / ~10% total from each dimension.               |
| `format_convert`        | Container swap: PNG→JPEG-95 or JPEG→PNG (direction from source).            |
| `screenshot_rasterise`  | Paste onto a neutral-grey canvas and re-encode as PNG (screenshot proxy).   |
| `brightness_mild`       | Pillow brightness enhance ×1.1, JPEG-95 output.                             |

Each transformation is deterministic (unit-tested) and produces bytes,
not paths — the runner writes them into the temp derivative directory,
runs every signal, then cleans up.

## Ground truth

- **SHA-256** — the source digest is registered in a temporary
  registry inside `tempfile.TemporaryDirectory` (so the user's real
  registry is never touched). Expected: `exact_match` on
  `original_copy`; `no_match` everywhere else.
- **C2PA** — the validator runs on the untouched source and its status
  is recorded as the per-image baseline. Expected: baseline verbatim
  on `original_copy`; `absent` after `metadata_stripped`; `None`
  (inconclusive) elsewhere — pixel changes may legitimately drop the
  manifest or invalidate it, and we do not fabricate a hard "expected"
  for those.
- **TrustMark** — the harness builds `max_trustmark_positives`
  watermarked derivatives at runtime using a fixed test payload
  (`ROBUST_EXP_2026`) with the requested variant (default `Q`) and
  treats them as the positive cohort. The unmodified source images
  form the unwatermarked-control cohort. Expected: `detected` on the
  positive `original_copy`; `not_detected` on every control row;
  `None` (inconclusive) on positive rows after non-control
  transformations — the measurement, not a pass/fail.

Watermarked positives are **never** relabelled as AI-generated. The
cohort column in the CSV keeps content origin and experimental
watermark presence strictly separate.

## Metrics

Full arithmetic lives in [`metrics.py`](metrics.py). One row per
`(image, transformation, signal)`; the runner writes the row list to
`detailed_results.csv` verbatim and computes per-signal summaries into
`summary_results.json`:

- SHA-256 exact-match rate by transformation.
- C2PA status breakdown by transformation (with a `valid_survival_rate`
  restricted to images whose baseline validated).
- TrustMark TP rate / FP rate / survival rate by transformation.
- Coverage per signal (fraction of rows the detector was able to
  answer at all).
- Inconclusive rate per signal.
- Mean / median wall-clock runtime per signal.

There is deliberately no combined "accuracy" number — mixing byte-
exactness, crypto-validation, and watermark decoding into one figure
would be misleading.

## Known limitations

- The default sample set is tiny (12 images). It is enough to
  demonstrate behaviour but not enough to make statistical claims —
  use `--input-dir` with a larger cohort for those.
- The `screenshot_rasterise` transformation is a mild proxy; a real
  screenshot also involves a viewer's chrome, subpixel rendering, and
  a colour-space round-trip.
- TrustMark's own robustness envelope is a property of the upstream
  library; a version bump can change survival numbers. The runner
  records the loaded variant in the config so a change is visible.
- The C2PA `expected_status` model is intentionally cautious:
  `inconclusive` for non-control, non-strip transformations. The raw
  detector output is still in the CSV — a stricter interpretation can
  be applied post-hoc without re-running.
- `data/sample_images/` currently contains no images with a real C2PA
  manifest, so the "valid survival" cohort is empty in the default run
  (the JSON reports that as `null` / an empty cohort — never as
  100%). Pointing `--input-dir` at C2PA-signed images makes that
  cohort populate.

## Reproduction

```bash
pip install -r requirements.txt
brew install exiftool                 # C2PA validator uses exiftool for some formats
pip install trustmark                 # if not already installed

python -m src.genai_detection.evaluation.robustness_experiments \
    --input-dir data/sample_images \
    --output-dir outputs/robustness \
    --seed 42
```

A pre-flight smoke run on the first two images with the minimal
transformation set:

```bash
python -m src.genai_detection.evaluation.robustness_experiments --smoke
```

Outputs go to `outputs/robustness/`:

```
outputs/robustness/
├── experiment_config.json      # exact CLI + transformation parameters
├── detailed_results.csv        # one row per (image, transformation, signal)
├── summary_results.json        # aggregated metrics
├── robustness_by_transformation.png
├── signal_coverage.png
└── README.md                   # this file (kept in the source tree, not copied)
```

If TrustMark's library or model weights cannot be loaded, the runner
still completes the C2PA and SHA-256 signals and records every
TrustMark row as `detector_unavailable`. That "the question could not
be answered" is a first-class outcome, distinct from
`not_detected`.
