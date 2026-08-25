"""
Robustness experiment runner for the three transparency signals.

Question this harness asks — and only this: **for each of the three
public, supported mechanisms this project ships, how does its output
change when the image is passed through a common transformation?**

The three signals:

1. **C2PA / metadata provenance** — cryptographic manifest validation
   via the official ``c2pa-python`` library (see
   :mod:`src.genai_detection.metadata_module.provenance_validation`).
2. **Adobe TrustMark** — the only invisible watermark scheme the pipeline
   supports (see
   :mod:`src.genai_detection.watermark_module.trustmark_detector`).
3. **SHA-256 byte-exact hash** — registry lookup against a small
   text-only table of known-file digests (see
   :mod:`src.genai_detection.hash_module`).

What this harness deliberately does NOT do:

* It does not evaluate SynthID or any arbitrary proprietary watermark —
  the pipeline has no detector for them and inventing a "not-found"
  status for a scheme we can't test would be dishonest.
* It does not produce an "overall accuracy" number that mixes the
  three signals. They answer different questions.
* It does not decide whether an image is AI-generated. A TrustMark or a
  hash registration is orthogonal to content origin — a TrustMark added
  at experiment time only proves we added a TrustMark.

Storage discipline: every transformed image lives inside a
:class:`tempfile.TemporaryDirectory` that is removed on exit (both on
success and on failure). Only compact CSV + JSON + PNG results are kept
on disk. The runner never writes into ``data/`` and never commits
generated variants — check ``git status`` after a run and there should
be nothing under ``data/``.

Reproduction (default location, mirrors the CLI in this file's docstring):

    python -m src.genai_detection.evaluation.robustness_experiments \\
        --input-dir data/sample_images \\
        --output-dir outputs/robustness \\
        --seed 42

Add ``--smoke`` to run on one or two images with the minimal
transformation set — the intended pre-flight before the full run.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT
from src.genai_detection.hash_module import (
    HashRecord,
    HashRegistry,
    OriginLabel,
    sha256_bytes,
    sha256_file,
)
from src.genai_detection.metadata_module import (
    OriginClaim,
    ProvenanceStatus,
    validate_provenance,
)
from src.genai_detection.watermark_module import (
    DEFAULT_VARIANT,
    SUPPORTED_VARIANTS,
    TrustMarkDetector,
    TrustMarkStatus,
)

from .metrics import (
    RESULT_COLUMNS,
    classify_outcome,
    summarise,
)
from .transformations import (
    SMOKE_TRANSFORMATIONS,
    TRANSFORMATIONS,
    transformations_config,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "robustness"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "sample_images"
DEFAULT_SAMPLE_LIMIT = 6
SMOKE_SAMPLE_LIMIT = 2

TRUSTMARK_TEST_PAYLOAD = "ROBUST_EXP_2026"
"""Fixed short ASCII payload embedded in every experiment-time TrustMark
derivative. Kept in one place so the CSV notes and the config file both
name the same value — the reader can verify the payload independently by
re-running the detector on any produced positive."""


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceImage:
    """One image the experiment will run over.

    Kept minimal on purpose — the transformation loop only needs the
    identifier (used to build the CSV ``image_id``), the on-disk path
    (used to re-read the bytes if the derivative loop needs them again),
    and the pre-computed source SHA-256 (used as the "byte-preserving
    control expected match" per row).
    """

    image_id: str
    path: Path
    sha256: str
    ext: str  # includes the leading dot


def _discover_images(input_dir: Path, limit: int | None) -> list[SourceImage]:
    """Enumerate image files under ``input_dir`` and hash each once.

    The runner intentionally sticks to a small, alphabetically-ordered
    subset so the default smoke run stays fast. Users can point
    ``--input-dir`` at their own directory to swap in a larger cohort.
    """
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if limit is not None:
        files = files[:limit]
    out: list[SourceImage] = []
    for p in files:
        out.append(
            SourceImage(
                image_id=p.name,
                path=p.resolve(),
                sha256=sha256_file(p),
                ext=p.suffix,
            )
        )
    return out


# ---------------------------------------------------------------------------
# TrustMark cohort construction
# ---------------------------------------------------------------------------


@dataclass
class _TrustMarkCohort:
    """Watermarked-positive derivatives created at experiment time.

    A per-source ``bytes`` blob carrying the TrustMark payload. Held in
    memory (never written to ``data/``) and only serialised into the
    per-image transformation temp directory when a row needs it.
    """

    variant: str
    payload: str
    positives: dict[str, bytes] = field(default_factory=dict)
    """Keyed by source ``image_id``; empty when the detector is
    unavailable or embedding failed on every image (in which case the
    TrustMark rows are reported as detector-unavailable and never mislead
    the summary)."""

    embedder_available: bool = False
    unavailable_reason: str | None = None


def _build_trustmark_cohort(
    sources: list[SourceImage],
    max_positives: int,
    variant: str,
) -> _TrustMarkCohort:
    """Embed TrustMark into the first ``max_positives`` source images.

    Uses the official ``trustmark.TrustMark(...).encode(...)`` API and
    keeps the payload deliberately short so the schema-version choice
    (currently upstream-default BCH_SUPER) can carry it. If the library
    or its weights are missing, returns a cohort with
    ``embedder_available=False`` — the runner still emits TrustMark rows
    marked ``detector_unavailable`` so the CSV shape is stable.
    """
    cohort = _TrustMarkCohort(variant=variant, payload=TRUSTMARK_TEST_PAYLOAD)

    try:
        from trustmark import TrustMark  # type: ignore[import-not-found]
    except Exception as exc:
        cohort.unavailable_reason = (
            f"trustmark library not importable: {type(exc).__name__}: {exc}"
        )
        return cohort

    try:
        embedder = TrustMark(
            model_type=variant,
            device="cpu",
            verbose=False,
            # Match the detector wrapper's opt-outs — no bbox / remover.
            loadBBoxDetector=False,
            loadRemover=False,
        )
    except Exception as exc:
        cohort.unavailable_reason = (
            f"TrustMark({variant!r}) failed to initialise: "
            f"{type(exc).__name__}: {exc}"
        )
        return cohort

    from PIL import Image

    for src in sources[:max_positives]:
        try:
            img = Image.open(src.path).convert("RGB")
            watermarked = embedder.encode(img, TRUSTMARK_TEST_PAYLOAD)
            buf = io.BytesIO()
            # PNG so the embed round-trips losslessly for the control
            # row; a JPEG re-encode is a separate transformation entry
            # that we WANT to measure survival through.
            watermarked.save(buf, format="PNG", optimize=False)
            cohort.positives[src.image_id] = buf.getvalue()
        except Exception as exc:
            # A per-image failure doesn't disable the cohort — record
            # the reason on stderr and move on, so a bad image can't
            # sink the whole run.
            print(
                f"  [trustmark cohort] embed failed for {src.image_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    cohort.embedder_available = bool(cohort.positives)
    if not cohort.embedder_available and cohort.unavailable_reason is None:
        cohort.unavailable_reason = (
            "TrustMark embedder loaded but every image failed to embed."
        )
    return cohort


# ---------------------------------------------------------------------------
# Ground-truth baseline
# ---------------------------------------------------------------------------


def _baseline_c2pa(source_path: Path) -> dict[str, Any]:
    """Run C2PA validation once on the source so per-row expectations
    reflect what the file actually carried before transformation.

    Returns a small dict with ``status`` (the raw enum value) and
    ``origin`` (the origin-claim value). Used by the runner to set
    ``expected_status`` on the ``original_copy`` row.
    """
    result = validate_provenance(source_path)
    return {
        "status": result.status.value,
        "origin": result.origin_claim.value,
        "validator_available": result.status is not ProvenanceStatus.VALIDATOR_UNAVAILABLE,
    }


# ---------------------------------------------------------------------------
# Per-signal evaluation
# ---------------------------------------------------------------------------


def _eval_sha256(
    transformed_bytes: bytes,
    registry: HashRegistry,
    src: SourceImage,
) -> dict[str, Any]:
    """One SHA-256 row.

    Always covered (the hasher and the in-memory registry cannot become
    "unavailable" mid-run). The expected outcome is a per-transformation
    property — the byte-preserving control must match, and every other
    transformation must not.
    """
    start = time.perf_counter()
    digest = sha256_bytes(transformed_bytes)
    lookup = registry.lookup(digest)
    elapsed = time.perf_counter() - start
    return {
        "signal": "sha256",
        "detector_available": True,
        "status": lookup.status.value,
        "runtime_seconds": elapsed,
        "error_details": lookup.error_details or "",
        "sha256_transformed": digest,
        "sha256_exact_match": digest == src.sha256,
    }


def _eval_c2pa(
    transformed_path: Path,
) -> dict[str, Any]:
    """One C2PA validation row.

    The validator only accepts a path, so the runner has already
    written ``transformed_bytes`` to a temp file. Every failure mode
    returns a typed status — this function never raises.
    """
    start = time.perf_counter()
    result = validate_provenance(transformed_path)
    elapsed = time.perf_counter() - start
    available = result.status is not ProvenanceStatus.VALIDATOR_UNAVAILABLE
    return {
        "signal": "c2pa",
        "detector_available": available,
        "status": result.status.value,
        "runtime_seconds": elapsed,
        "error_details": "; ".join(result.validation_errors) if result.validation_errors else "",
        "c2pa_origin_claim": result.origin_claim.value,
    }


def _eval_trustmark(
    transformed_bytes: bytes,
    detector: TrustMarkDetector,
    variant: str,
) -> dict[str, Any]:
    """One TrustMark row. The detector takes bytes directly."""
    result = detector.analyse(transformed_bytes, variant=variant)
    return {
        "signal": "trustmark",
        "detector_available": result.status is not TrustMarkStatus.DETECTOR_UNAVAILABLE,
        "status": result.status.value,
        "runtime_seconds": result.processing_time_seconds,
        "error_details": result.error_details or "",
        "trustmark_variant": result.variant_used or "",
        "trustmark_schema_version": (
            "" if result.schema_version is None else str(result.schema_version)
        ),
    }


def _trustmark_unavailable_row(
    variant: str,
    reason: str,
) -> dict[str, Any]:
    """Placeholder TrustMark row for use when the library/model is
    missing — keeps the CSV shape stable so the summary can honestly
    report coverage."""
    return {
        "signal": "trustmark",
        "detector_available": False,
        "status": TrustMarkStatus.DETECTOR_UNAVAILABLE.value,
        "runtime_seconds": 0.0,
        "error_details": reason,
        "trustmark_variant": variant,
        "trustmark_schema_version": "",
    }


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def _sha256_expected(transformation_name: str) -> str:
    """Only the byte-preserving copy is expected to match. Everything
    else changes at least one byte and so must miss."""
    return "exact_match" if transformation_name == "original_copy" else "no_match"


def _c2pa_expected(
    transformation_name: str,
    baseline_status: str,
) -> str | None:
    """C2PA expectation per transformation.

    * ``original_copy`` — expected to match the baseline verbatim.
    * ``metadata_stripped`` — expected ``absent`` (the transformation
      re-encodes without an EXIF/XMP block, which also drops any C2PA
      manifest embedded in metadata).
    * Every other transformation is honest about being non-judgemental —
      the outcome is ``inconclusive`` when the baseline was already
      ``absent`` (nothing to preserve), and ``inconclusive`` after a
      re-encode / geometric change too, because C2PA may legitimately
      end up ``absent`` or ``invalid_or_tampered`` after a pixel change
      and we don't want to fabricate a hard "expected" for that.

    Returning ``None`` marks the row as inconclusive on purpose — see
    :func:`~.metrics.classify_outcome`.
    """
    if transformation_name == "original_copy":
        return baseline_status
    if transformation_name == "metadata_stripped":
        return "absent"
    return None


def _trustmark_expected(
    cohort: str,
    transformation_name: str,
) -> str | None:
    """TrustMark expectation.

    * ``unwatermarked_control`` cohort: every row is expected to read
      ``not_detected``. This is the false-positive check.
    * ``watermarked_positive`` cohort: only ``original_copy`` is
      expected to read ``detected``. Other transformations are the
      survival measurement — no "correct" answer, so we return
      ``None`` and record whatever the detector reports.
    """
    if cohort == "unwatermarked_control":
        return "not_detected"
    if cohort == "watermarked_positive" and transformation_name == "original_copy":
        return "detected"
    return None


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Structured view of the CLI arguments, serialised into
    ``experiment_config.json``. Contains only JSON-safe types."""

    input_dir: str
    output_dir: str
    seed: int
    smoke: bool
    sample_limit: int
    transformations: list[str]
    trustmark_variant: str
    trustmark_payload: str
    max_trustmark_positives: int


def _run_experiment(
    sources: list[SourceImage],
    transformation_names: list[str],
    output_dir: Path,
    trustmark_variant: str,
    max_trustmark_positives: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Iterate over (image, transformation) pairs and evaluate every
    signal on each. Returns the raw row list plus a small run summary
    (paths, cohort sizes, transformation list) the runner adds to the
    final JSON."""

    # SHA-256 registry lives entirely in a temp path so this experiment
    # never touches the user's live registry. ``HashRegistry`` requires
    # the file to exist before construction, so we seed it with an
    # empty envelope first.
    tmp_registry_dir = Path(tempfile.mkdtemp(prefix="robust_registry_"))
    registry_path = tmp_registry_dir / "registry.json"
    registry_path.write_text('{"records": []}', encoding="utf-8")
    registry = HashRegistry(registry_path)
    for src in sources:
        registry.register(
            HashRecord(
                sha256=src.sha256,
                origin_label=OriginLabel.UNKNOWN,
                provider="robustness_experiment",
                model=None,
                source_reference=str(src.image_id),
                notes="Registered at experiment time; origin label is unknown by design.",
            )
        )

    # TrustMark detector wraps the official library. Instantiation is
    # cheap; the first analyse() call triggers the weight download.
    trustmark_detector = TrustMarkDetector(default_variant=trustmark_variant)

    # Positive cohort — created only if the library is available.
    positives = _build_trustmark_cohort(
        sources,
        max_positives=max_trustmark_positives,
        variant=trustmark_variant,
    )
    if positives.embedder_available:
        print(f"  [trustmark] embedded positives: {len(positives.positives)} images")
    else:
        print(f"  [trustmark] positives unavailable: {positives.unavailable_reason}")

    # Baseline C2PA once per source — used to set expectations on the
    # original_copy row without re-running validation per transformation.
    baselines: dict[str, dict[str, Any]] = {
        src.image_id: _baseline_c2pa(src.path) for src in sources
    }

    rows: list[dict[str, Any]] = []

    # The temp directory holding every derivative bytes-on-disk. Wrapped
    # in a try/finally so we clean up even on error partway through the
    # loop — the storage rule is "no derivative survives the run".
    tmp_derivatives_dir = Path(tempfile.mkdtemp(prefix="robust_derivs_"))
    try:
        for src in sources:
            src_bytes = src.path.read_bytes()

            # Every non-TrustMark row runs over the "content" cohort:
            # C2PA validation and SHA-256 lookup are properties of the
            # source pixels/bytes and are unrelated to whether an
            # experimental TrustMark was added.
            for tf_name in transformation_names:
                tf = TRANSFORMATIONS[tf_name]
                try:
                    out_bytes, out_suffix = tf.apply(src_bytes, src.ext)
                except Exception as exc:
                    # A broken transformation shows up as a single ERROR
                    # row for every signal so the reader can spot it.
                    err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    for signal in ("sha256", "c2pa", "trustmark"):
                        rows.append(
                            _build_row(
                                src=src,
                                cohort="content",
                                tf_name=tf_name,
                                tf_params=tf.params,
                                signal_row={
                                    "signal": signal,
                                    "detector_available": True,
                                    "status": "error",
                                    "runtime_seconds": 0.0,
                                    "error_details": err,
                                },
                                expected=None,
                            )
                        )
                    continue

                # Write once — the C2PA validator wants a path.
                deriv_path = tmp_derivatives_dir / f"{src.image_id}__{tf_name}{out_suffix}"
                deriv_path.write_bytes(out_bytes)

                # --- SHA-256 ---
                sha_row = _eval_sha256(out_bytes, registry, src)
                sha_row.setdefault("sha256_source", src.sha256)
                rows.append(
                    _build_row(
                        src=src,
                        cohort="content",
                        tf_name=tf_name,
                        tf_params=tf.params,
                        signal_row=sha_row,
                        expected=_sha256_expected(tf_name),
                    )
                )

                # --- C2PA ---
                c2pa_row = _eval_c2pa(deriv_path)
                rows.append(
                    _build_row(
                        src=src,
                        cohort="content",
                        tf_name=tf_name,
                        tf_params=tf.params,
                        signal_row=c2pa_row,
                        expected=_c2pa_expected(tf_name, baselines[src.image_id]["status"]),
                    )
                )

                # --- TrustMark, unwatermarked-control cohort ---
                # Every source image goes into the false-positive
                # cohort — we never added a watermark, we don't want to
                # find one.
                if positives.embedder_available:
                    tm_row = _eval_trustmark(out_bytes, trustmark_detector, trustmark_variant)
                else:
                    tm_row = _trustmark_unavailable_row(
                        trustmark_variant, positives.unavailable_reason or ""
                    )
                rows.append(
                    _build_row(
                        src=src,
                        cohort="unwatermarked_control",
                        tf_name=tf_name,
                        tf_params=tf.params,
                        signal_row=tm_row,
                        expected=_trustmark_expected("unwatermarked_control", tf_name),
                    )
                )

            # TrustMark positives are a separate cohort — each is a
            # DIFFERENT image (the source with a watermark embedded).
            # The transformation still runs over the watermarked bytes.
            wm_bytes = positives.positives.get(src.image_id)
            if wm_bytes is None:
                continue
            for tf_name in transformation_names:
                tf = TRANSFORMATIONS[tf_name]
                try:
                    out_bytes, _out_suffix = tf.apply(wm_bytes, ".png")
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    rows.append(
                        _build_row(
                            src=src,
                            cohort="watermarked_positive",
                            tf_name=tf_name,
                            tf_params=tf.params,
                            signal_row={
                                "signal": "trustmark",
                                "detector_available": True,
                                "status": "error",
                                "runtime_seconds": 0.0,
                                "error_details": err,
                            },
                            expected=_trustmark_expected("watermarked_positive", tf_name),
                        )
                    )
                    continue
                tm_row = _eval_trustmark(out_bytes, trustmark_detector, trustmark_variant)
                rows.append(
                    _build_row(
                        src=src,
                        cohort="watermarked_positive",
                        tf_name=tf_name,
                        tf_params=tf.params,
                        signal_row=tm_row,
                        expected=_trustmark_expected("watermarked_positive", tf_name),
                    )
                )
    finally:
        # Belt-and-braces: every derivative file goes away, even on error.
        shutil.rmtree(tmp_derivatives_dir, ignore_errors=True)
        shutil.rmtree(tmp_registry_dir, ignore_errors=True)

    run_meta = {
        "source_count": len(sources),
        "transformation_count": len(transformation_names),
        "trustmark_available": positives.embedder_available,
        "trustmark_unavailable_reason": positives.unavailable_reason,
        "trustmark_positive_count": len(positives.positives),
        "baselines": baselines,
    }
    return rows, run_meta


def _build_row(
    src: SourceImage,
    cohort: str,
    tf_name: str,
    tf_params: dict,
    signal_row: dict[str, Any],
    expected: str | None,
) -> dict[str, Any]:
    """Assemble one CSV row from a per-signal partial result.

    Kept small so the callers stay focused on the signal logic; every
    column in :data:`RESULT_COLUMNS` gets a default here so the CSV is
    dense even for the failure paths.
    """
    row: dict[str, Any] = {c: "" for c in RESULT_COLUMNS}
    row.update(
        {
            "image_id": src.image_id,
            "source_path": str(src.path),
            "cohort": cohort,
            "transformation": tf_name,
            "params_json": json.dumps(
                {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in tf_params.items()
                },
                sort_keys=True,
            ),
            "expected_status": expected or "",
            "sha256_source": src.sha256,
        }
    )
    row.update(signal_row)
    row["outcome"] = classify_outcome(row["status"], expected)
    return row


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RESULT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})


def _write_config(config: ExperimentConfig, transformation_specs: list[dict[str, Any]], path: Path) -> None:
    payload = {
        "input_dir": config.input_dir,
        "output_dir": config.output_dir,
        "seed": config.seed,
        "smoke": config.smoke,
        "sample_limit": config.sample_limit,
        "trustmark_variant": config.trustmark_variant,
        "trustmark_payload": config.trustmark_payload,
        "max_trustmark_positives": config.max_trustmark_positives,
        "transformations": transformation_specs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_summary(summary: dict[str, Any], run_meta: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps({"summary": summary, "run": run_meta}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


_OUTPUT_README_TEMPLATE = """# Robustness experiment — outputs

Regenerated on every run of
`python -m src.genai_detection.evaluation.robustness_experiments`.

## Files

- `experiment_config.json` — exact CLI arguments plus the full spec of
  every transformation used, including its parameters. Enough to
  rebuild every derivative from the source images alone.
- `detailed_results.csv` — one row per `(image, transformation,
  signal)`. Columns are documented in
  [`../../src/genai_detection/evaluation/metrics.py`](../../src/genai_detection/evaluation/metrics.py)
  (`RESULT_COLUMNS`).
- `summary_results.json` — aggregated metrics computed from
  `detailed_results.csv`. Each of the three signals (SHA-256, C2PA,
  TrustMark) is reported separately by design — there is no combined
  "accuracy" number.
- `robustness_by_transformation.png` — grouped bars, one bar per
  signal per transformation. SHA-256 = byte-exact match rate; C2PA =
  `valid` survival rate on the cohort whose original validated;
  TrustMark = watermarked-positive survival rate.
- `signal_coverage.png` — coverage vs. inconclusive rate per signal.

## Labels used

- `Detected` / `Not detected` — TrustMark outcomes.
- `Valid` / `Absent` / `Invalid` / `Untrusted signer` — C2PA outcomes.
- `Exact match` / `No match` — SHA-256 outcomes.
- `Unavailable` — the detector or its dependency was missing / errored.
  Never conflated with a negative detection.
- `Inconclusive` — the row has no ground truth to score against
  (e.g. TrustMark survival on a non-control transformation).

`Real` is never used to describe an absent transparency signal —
missing metadata, a missing watermark, or a missing registry entry all
mean only "this signal could not attest anything", not "the image is
human-authored".

## Reproduction

See the harness README at
[`../../src/genai_detection/evaluation/README.md`](../../src/genai_detection/evaluation/README.md).

## Scope

Evaluates the mechanisms this project actually implements. Does not
evaluate SynthID or arbitrary proprietary watermarks; those return
`Unavailable` on the TrustMark row rather than a fabricated
`Not detected`.
"""


def _write_output_readme(path: Path) -> None:
    """Small README dropped next to the results so the outputs are
    self-describing when opened outside the repo."""
    path.write_text(_OUTPUT_README_TEMPLATE, encoding="utf-8")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_robustness_by_transformation(summary: dict[str, Any], path: Path) -> None:
    """One grouped bar chart: rows are transformations, groups are signals.

    Rendered with Agg so the harness works in a headless run. Label
    conventions ("Detected", "Not detected", "Absent" …) come from the
    module contracts — the plot never uses "Real" for a missing signal.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tfs = list(TRANSFORMATIONS)  # order matches the registry
    active_tfs = [
        t for t in tfs
        if t in summary["sha256_exact_match_by_transformation"]
        or t in summary["c2pa_by_transformation"]
        or t in summary["trustmark_by_transformation"]
    ]

    sha_vals = [
        _safe_rate(summary["sha256_exact_match_by_transformation"].get(t, {}).get("rate"))
        for t in active_tfs
    ]
    c2pa_vals = [
        _safe_rate(summary["c2pa_by_transformation"].get(t, {}).get("valid_survival_rate"))
        for t in active_tfs
    ]
    tm_vals = [
        _safe_rate(summary["trustmark_by_transformation"].get(t, {}).get("survival_rate"))
        for t in active_tfs
    ]

    import numpy as np
    x = np.arange(len(active_tfs))
    width = 0.28
    fig, ax = plt.subplots(figsize=(max(6, len(active_tfs) * 0.9), 4.5))
    ax.bar(x - width, sha_vals, width, label="SHA-256 exact match", color="#4c72b0")
    ax.bar(x, c2pa_vals, width, label="C2PA valid (survival)", color="#dd8452")
    ax.bar(x + width, tm_vals, width, label="TrustMark detected (survival)", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(active_tfs, rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Signal survival by transformation")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_signal_coverage(summary: dict[str, Any], path: Path) -> None:
    """Coverage bar chart per signal — did the detector answer at all?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals = list(summary["coverage_by_signal"])
    rates = [
        _safe_rate(summary["coverage_by_signal"][s]["rate"])
        for s in signals
    ]
    inconclusive = [
        _safe_rate(summary["inconclusive_rate"].get(s, {}).get("rate"))
        for s in signals
    ]

    import numpy as np
    x = np.arange(len(signals))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(x - width / 2, rates, width, label="Covered", color="#4c72b0")
    ax.bar(x + width / 2, inconclusive, width, label="Inconclusive", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() if s == "sha256" else s.capitalize() for s in signals])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Signal coverage vs. inconclusive rate")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _safe_rate(v: Any) -> float:
    """Turn ``None`` / missing rates into a 0-height bar without hiding
    the fact — the summary JSON keeps the true ``None``, the plot just
    can't render one."""
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.genai_detection.evaluation.robustness_experiments",
        description=(
            "Reproducible robustness comparison of the three implemented "
            "transparency signals (C2PA, TrustMark, SHA-256) under common "
            "image transformations."
        ),
    )
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                   help="Directory of source images (default: data/sample_images).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Where to write CSV / JSON / PNG results (default: outputs/robustness).")
    p.add_argument("--seed", type=int, default=42,
                   help="Recorded in the config for reproducibility (no random sampling happens).")
    p.add_argument("--smoke", action="store_true",
                   help="Run on the first %d images with a minimal transformation set." % SMOKE_SAMPLE_LIMIT)
    p.add_argument("--sample-limit", type=int, default=None,
                   help="Cap the number of source images (default: %d, or %d with --smoke)."
                        % (DEFAULT_SAMPLE_LIMIT, SMOKE_SAMPLE_LIMIT))
    p.add_argument("--trustmark-variant", type=str, default=DEFAULT_VARIANT,
                   choices=list(SUPPORTED_VARIANTS),
                   help="TrustMark model_type to embed and decode (default: %s)." % DEFAULT_VARIANT)
    p.add_argument("--max-trustmark-positives", type=int, default=2,
                   help="Number of watermarked-positive derivatives to build.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_limit = (
        args.sample_limit
        if args.sample_limit is not None
        else (SMOKE_SAMPLE_LIMIT if args.smoke else DEFAULT_SAMPLE_LIMIT)
    )
    tf_names = list(SMOKE_TRANSFORMATIONS) if args.smoke else list(TRANSFORMATIONS)

    # Discover sources first — we want a clear failure if the input dir
    # is empty rather than an empty CSV.
    sources = _discover_images(args.input_dir, sample_limit)
    if not sources:
        print(f"No image files found under {args.input_dir}", file=sys.stderr)
        return 2

    print(f"Robustness experiment — {len(sources)} images × {len(tf_names)} transformations")
    print(f"  input : {args.input_dir}")
    print(f"  output: {output_dir}")

    config = ExperimentConfig(
        input_dir=str(args.input_dir),
        output_dir=str(output_dir),
        seed=int(args.seed),
        smoke=bool(args.smoke),
        sample_limit=sample_limit,
        transformations=tf_names,
        trustmark_variant=args.trustmark_variant,
        trustmark_payload=TRUSTMARK_TEST_PAYLOAD,
        max_trustmark_positives=int(args.max_trustmark_positives),
    )
    _write_config(config, transformations_config(tf_names), output_dir / "experiment_config.json")

    rows, run_meta = _run_experiment(
        sources=sources,
        transformation_names=tf_names,
        output_dir=output_dir,
        trustmark_variant=args.trustmark_variant,
        max_trustmark_positives=int(args.max_trustmark_positives),
    )

    csv_path = output_dir / "detailed_results.csv"
    _write_csv(rows, csv_path)
    summary = summarise(rows)
    summary_path = output_dir / "summary_results.json"
    _write_summary(summary, run_meta, summary_path)

    robustness_png = output_dir / "robustness_by_transformation.png"
    coverage_png = output_dir / "signal_coverage.png"
    try:
        _plot_robustness_by_transformation(summary, robustness_png)
        _plot_signal_coverage(summary, coverage_png)
    except Exception as exc:  # pragma: no cover - plotting failure is not fatal
        print(f"  [plot] failed to render figures: {type(exc).__name__}: {exc}", file=sys.stderr)

    _write_output_readme(output_dir / "README.md")

    # A tiny "what did we produce" print so the user gets a useful
    # terminal summary without opening the JSON.
    _print_summary(rows, summary, run_meta, output_dir)
    return 0


def _print_summary(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    run_meta: dict[str, Any],
    output_dir: Path,
) -> None:
    print("\nExperiment complete.")
    print(f"  rows       : {len(rows)}")
    print(f"  csv        : {output_dir / 'detailed_results.csv'}")
    print(f"  summary    : {output_dir / 'summary_results.json'}")
    print(f"  figures    : {output_dir / 'robustness_by_transformation.png'}, "
          f"{output_dir / 'signal_coverage.png'}")
    print("  coverage_by_signal:")
    for signal, stats in summary["coverage_by_signal"].items():
        rate = stats["rate"]
        pct = "n/a" if rate is None else f"{rate * 100:.1f}%"
        print(f"    {signal:<10} covered={stats['covered']}/{stats['total']} ({pct})")
    if not run_meta.get("trustmark_available", False):
        print(f"  trustmark  : unavailable — {run_meta.get('trustmark_unavailable_reason')}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
