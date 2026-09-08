"""
Visual-classifier robustness experiment — extends the three-signal harness
in ``robustness_experiments.py`` with the CANONICAL Community Forensics 384
detector.

Design (matches the P1 spec in METHODOLOGY_EVALUATION_GUIDE.md §14):

* Cohort: balanced 50/50 real/AI sampled evenly from the in-distribution
  ``combined_dataset/test`` split (the same split ``run_07…`` scored). The
  split lives locally at ``data/visual/combined_dataset/test`` — no HF
  download required.
* Transformations: the SAME 12 defined in
  :mod:`.transformations` (byte-preserving control + 11 alterations).
* Baseline: every image is scored once at ``original_copy`` — its P(AI) on
  the untransformed bytes. Every other row records the transformed P(AI)
  AND its delta versus the baseline for that same image.
* Metrics reported per transformation (see :mod:`.metrics` for the boring
  aggregation helpers):
    - accuracy, precision, recall, F1 at the canonical fused decision
      threshold (``config.WA_DECISION_THRESHOLD``, currently 0.16 per the
      2026-09-06 sweep — historical 0.55 outputs are archived under
      ``outputs/robustness_visual_classifier_thr055_historical/``)
    - ROC-AUC (undefined when a transformation leaves only one class)
    - mean / median / IQR of ΔP(AI) vs baseline
    - classification flip rate: fraction of images whose decision at the
      canonical threshold changes vs baseline
    - fraction of decisions that changed toward "AI" vs toward "Real"
* Writes to a NEW, versioned output directory
  (``outputs/robustness_visual_classifier/``) — never touches the existing
  ``outputs/robustness/`` bundle.

Runtime rule: the CF-384 model runs on MPS on Apple silicon (roughly one
inference per 100 ms on a batch of 1, ~50 ms in a small batch), so a cohort
of ~150 images × 12 transformations × 1 baseline stays well under 10 min.

CLI mirrors robustness_experiments.py:

    python -m src.genai_detection.evaluation.visual_classifier_robustness \\
        --sample-size 150 --seed 42

The seed is threaded through to the numpy sampler that picks image indices,
so re-running yields the same cohort.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src import PROJECT_ROOT
from src.genai_detection.integration_pipeline import config
from src.genai_detection.visual_module.evaluation import iter_split_images
from src.genai_detection.visual_module.visual_classifier import (
    COMMFOR_MODEL_384,
    CommunityForensicsClassifier,
)

from .transformations import TRANSFORMATIONS


DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "visual" / "combined_dataset" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "robustness_visual_classifier"

# Label convention in combined_dataset: 1 == AI-generated, 0 == real
# (matches iter_split_images and score_split in evaluation.py).
LABEL_AI = 1
LABEL_REAL = 0


@dataclass
class Sample:
    """One image drawn from the local combined_dataset test split."""

    index: int  # source-order index (streaming rank)
    label: int  # 0 real, 1 AI
    ext: str
    bytes: bytes  # PNG-encoded (lossless) for downstream transformations


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    """Encode a PIL image as lossless PNG bytes.

    We need bytes (not a live PIL image) because every transformation in
    ``transformations.py`` expects the ``(src_bytes, src_suffix, params)``
    contract. PNG keeps the pixels exact, so the ``original_copy`` row
    genuinely round-trips the source pixels the classifier scored.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _sample_balanced_cohort(
    split_dir: Path,
    sample_size: int,
    seed: int,
) -> list[Sample]:
    """
    Deterministically pick ``sample_size`` images from ``split_dir``, balanced
    across labels {0, 1}.

    Strategy: stream the split index-by-label first, then take an evenly-
    strided subset of each class so the sample is not front-loaded with the
    first shard's images. This mirrors the strided sampling
    ``score_commfor_eval`` uses for the CF eval set (see evaluation.py).
    """
    per_class = sample_size // 2

    # Read labels first (cheap) and record streaming indices per class.
    reals: list[int] = []
    ais: list[int] = []
    total = 0
    for _idx, (_img, label) in enumerate(iter_split_images(str(split_dir))):
        if label == LABEL_REAL:
            reals.append(_idx)
        elif label == LABEL_AI:
            ais.append(_idx)
        total += 1

    if len(reals) < per_class or len(ais) < per_class:
        raise RuntimeError(
            f"Not enough images in split for balanced sampling: "
            f"reals={len(reals)}, ai={len(ais)}, need {per_class} each."
        )

    def _strided(indices: list[int], k: int) -> list[int]:
        step = len(indices) / k
        return [indices[int(round(i * step))] for i in range(k)]

    picked = sorted(set(_strided(reals, per_class) + _strided(ais, per_class)))

    # Second pass: read the actual image bytes for the picked indices.
    picked_set = set(picked)
    out: list[Sample] = []
    for _idx, (img, label) in enumerate(iter_split_images(str(split_dir))):
        if _idx in picked_set:
            out.append(
                Sample(index=_idx, label=int(label), ext=".png", bytes=_pil_to_png_bytes(img))
            )
            if len(out) == len(picked):
                break

    # Rank the sample deterministically for reproducibility.
    out.sort(key=lambda s: s.index)
    # `seed` is recorded for provenance; it does not currently randomise the
    # strided pick above (which is deterministic in `split_dir`).
    _ = seed
    return out


def _run_classifier_on_bytes(classifier, image_bytes: bytes) -> float:
    """Decode ``image_bytes`` and return P(AI) as a float."""
    pil = Image.open(io.BytesIO(image_bytes))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return float(classifier.ai_probabilities([pil])[0])


def _classify(prob: float, threshold: float) -> int:
    return int(prob >= threshold)


def _flag(baseline_pred: int, transformed_pred: int) -> str:
    if baseline_pred == transformed_pred:
        return "unchanged"
    if baseline_pred == 0 and transformed_pred == 1:
        return "flipped_to_ai"
    return "flipped_to_real"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _summary_per_transformation(rows: list[dict]) -> dict[str, Any]:
    """Compute per-transformation metrics from the row list.

    Kept in-file so it stays adjacent to the row schema — the aggregation
    logic in metrics.py is signal-agnostic (SHA-256 / C2PA / TrustMark) and
    was not designed to compute classification metrics with a ground-truth
    label column.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    per_tf: dict[str, list[dict]] = {}
    for row in rows:
        per_tf.setdefault(row["transformation"], []).append(row)

    out: dict[str, Any] = {}
    for tf, tf_rows in per_tf.items():
        labels = np.array([r["ground_truth"] for r in tf_rows], dtype=int)
        probs = np.array([r["transformed_ai_prob"] for r in tf_rows], dtype=float)
        preds = (probs >= config.WA_DECISION_THRESHOLD).astype(int)

        deltas = np.array(
            [r["ai_prob_delta"] for r in tf_rows if r["ai_prob_delta"] is not None],
            dtype=float,
        )
        flips = sum(1 for r in tf_rows if r["classification_changed"])
        flips_to_ai = sum(1 for r in tf_rows if r["flip_direction"] == "flipped_to_ai")
        flips_to_real = sum(1 for r in tf_rows if r["flip_direction"] == "flipped_to_real")

        both_classes = len(set(labels.tolist())) > 1
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        summary = {
            "n": int(len(tf_rows)),
            "n_real": int((labels == LABEL_REAL).sum()),
            "n_ai": int((labels == LABEL_AI).sum()),
            "threshold": config.WA_DECISION_THRESHOLD,
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(roc_auc_score(labels, probs)) if both_classes else None,
            "mean_p_ai": float(probs.mean()),
            "median_p_ai": float(np.median(probs)),
            "delta_p_ai_mean": float(deltas.mean()) if deltas.size else None,
            "delta_p_ai_median": float(np.median(deltas)) if deltas.size else None,
            "delta_p_ai_abs_mean": float(np.abs(deltas).mean()) if deltas.size else None,
            "delta_p_ai_abs_median": float(np.median(np.abs(deltas))) if deltas.size else None,
            "delta_p_ai_iqr": (
                float(np.percentile(deltas, 75) - np.percentile(deltas, 25))
                if deltas.size
                else None
            ),
            "classification_flip_rate": flips / len(tf_rows),
            "flips_to_ai": flips_to_ai,
            "flips_to_real": flips_to_real,
        }
        out[tf] = summary
    return out


def _write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _plot_summary(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf_names = list(TRANSFORMATIONS)
    order = [t for t in tf_names if t in summary]

    accs = [summary[t]["accuracy"] for t in order]
    f1s = [summary[t]["f1"] for t in order]
    flip_rates = [summary[t]["classification_flip_rate"] for t in order]
    delta_means = [summary[t]["delta_p_ai_mean"] or 0.0 for t in order]

    fig, axes = plt.subplots(2, 1, figsize=(max(7, len(order) * 0.9), 8))

    ax = axes[0]
    x = np.arange(len(order))
    width = 0.35
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4c72b0")
    ax.bar(x + width / 2, f1s, width, label="F1", color="#55a868")
    ax.plot(x, flip_rates, "o-", color="#c44e52", label="Classification flip rate")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Visual classifier (Community Forensics 384) — per-transformation")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    ax = axes[1]
    ax.bar(x, delta_means, color="#8172b2")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Mean ΔP(AI) vs baseline")
    ax.set_title("Signed drift in P(AI) after transformation")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_delta_boxplot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_tf: dict[str, list[float]] = {}
    for row in rows:
        if row["ai_prob_delta"] is None:
            continue
        per_tf.setdefault(row["transformation"], []).append(row["ai_prob_delta"])

    order = [t for t in TRANSFORMATIONS if t in per_tf and t != "original_copy"]
    data = [per_tf[t] for t in order]

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.9), 4.5))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("ΔP(AI) (transformed − baseline)")
    ax.set_title("Per-image ΔP(AI) distribution by transformation")
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


ROW_COLUMNS = [
    "sample_id",
    "ground_truth",
    "transformation",
    "params_json",
    "baseline_ai_prob",
    "transformed_ai_prob",
    "ai_prob_delta",
    "baseline_pred",
    "transformed_pred",
    "classification_changed",
    "flip_direction",
    "runtime_seconds",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.genai_detection.evaluation.visual_classifier_robustness",
        description=(
            "Robustness evaluation of the canonical Community Forensics 384 "
            "visual classifier across the same 12 transformations used by "
            "robustness_experiments.py."
        ),
    )
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sample-size", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model-id", type=str, default=COMMFOR_MODEL_384,
        help="HuggingFace repo of the CF checkpoint to score.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.sample_size} images from {args.split_dir} ...")
    t0 = time.perf_counter()
    samples = _sample_balanced_cohort(args.split_dir, args.sample_size, args.seed)
    print(
        f"  cohort: {len(samples)} images, "
        f"{sum(1 for s in samples if s.label == LABEL_REAL)} real / "
        f"{sum(1 for s in samples if s.label == LABEL_AI)} ai "
        f"(sample took {time.perf_counter() - t0:.1f}s)"
    )

    print(f"Loading visual classifier: {args.model_id}")
    classifier = CommunityForensicsClassifier(repo_id=args.model_id)

    tf_names = list(TRANSFORMATIONS)
    rows: list[dict] = []
    baseline_probs: dict[int, float] = {}
    baseline_preds: dict[int, int] = {}

    total_iters = len(samples) * len(tf_names)
    done = 0
    t_start = time.perf_counter()

    for sample in samples:
        for tf_name in tf_names:
            tf = TRANSFORMATIONS[tf_name]
            try:
                tf_bytes, _ext = tf.apply(sample.bytes, sample.ext)
            except Exception as exc:  # pragma: no cover — defensive
                rows.append(
                    {
                        "sample_id": sample.index,
                        "ground_truth": sample.label,
                        "transformation": tf_name,
                        "params_json": json.dumps(tf.params, sort_keys=True),
                        "baseline_ai_prob": baseline_probs.get(sample.index),
                        "transformed_ai_prob": None,
                        "ai_prob_delta": None,
                        "baseline_pred": baseline_preds.get(sample.index),
                        "transformed_pred": None,
                        "classification_changed": None,
                        "flip_direction": f"error:{type(exc).__name__}",
                        "runtime_seconds": 0.0,
                    }
                )
                done += 1
                continue

            ts = time.perf_counter()
            p_ai = _run_classifier_on_bytes(classifier, tf_bytes)
            elapsed = time.perf_counter() - ts
            pred = _classify(p_ai, config.WA_DECISION_THRESHOLD)

            if tf_name == "original_copy":
                baseline_probs[sample.index] = p_ai
                baseline_preds[sample.index] = pred

            baseline_p = baseline_probs.get(sample.index)
            baseline_pred = baseline_preds.get(sample.index)
            delta = None if baseline_p is None else (p_ai - baseline_p)
            changed = None if baseline_pred is None else (pred != baseline_pred)
            direction = _flag(baseline_pred, pred) if baseline_pred is not None else ""

            rows.append(
                {
                    "sample_id": sample.index,
                    "ground_truth": sample.label,
                    "transformation": tf_name,
                    "params_json": json.dumps(tf.params, sort_keys=True),
                    "baseline_ai_prob": baseline_p,
                    "transformed_ai_prob": p_ai,
                    "ai_prob_delta": delta,
                    "baseline_pred": baseline_pred,
                    "transformed_pred": pred,
                    "classification_changed": changed,
                    "flip_direction": direction,
                    "runtime_seconds": elapsed,
                }
            )
            done += 1
            if done % 100 == 0:
                elapsed_all = time.perf_counter() - t_start
                rate = done / elapsed_all if elapsed_all else 0
                eta = (total_iters - done) / rate if rate else float("inf")
                print(
                    f"  {done}/{total_iters} inferences "
                    f"({100 * done / total_iters:.1f}%), rate {rate:.1f}/s, ETA {eta:.0f}s",
                    flush=True,
                )

    # Sanity: original_copy must be an exact P(AI) baseline for every image
    # (the transformation is byte-preserving; the model is deterministic on
    # a single input at a fixed device). If any baseline is missing that is
    # a bug — refuse to save a summary with a hole in the delta column.
    missing = [s.index for s in samples if s.index not in baseline_probs]
    if missing:
        raise RuntimeError(
            f"Baseline P(AI) missing for {len(missing)} samples "
            f"(first 5: {missing[:5]}). Refusing to write summary."
        )

    summary = _summary_per_transformation(rows)
    summary_run = {
        "model_id": args.model_id,
        "split_dir": str(args.split_dir),
        "sample_size": len(samples),
        "seed": args.seed,
        "threshold": config.WA_DECISION_THRESHOLD,
        "transformations": tf_names,
        "row_count": len(rows),
        "wall_clock_seconds": time.perf_counter() - t_start,
    }

    _write_csv(rows, args.output_dir / "detailed_results.csv", ROW_COLUMNS)
    (args.output_dir / "summary_results.json").write_text(
        json.dumps({"run": summary_run, "summary": summary}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        _plot_summary(summary, args.output_dir / "accuracy_and_flip_by_transformation.png")
        _plot_delta_boxplot(rows, args.output_dir / "delta_p_ai_boxplot.png")
    except Exception as exc:  # pragma: no cover
        print(f"  [plot] failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    (args.output_dir / "README.md").write_text(
        (
            "# Visual-classifier robustness — outputs\n\n"
            "Regenerated on every run of "
            "`python -m src.genai_detection.evaluation.visual_classifier_robustness`.\n\n"
            "## Files\n\n"
            "- `detailed_results.csv` — one row per (image, transformation). "
            "Columns: sample_id, ground_truth (0=real, 1=AI), transformation, "
            "params_json, baseline_ai_prob, transformed_ai_prob, ai_prob_delta, "
            "baseline_pred, transformed_pred, classification_changed, flip_direction, "
            "runtime_seconds.\n"
            "- `summary_results.json` — per-transformation metrics: accuracy, "
            "precision, recall, F1, ROC-AUC, mean/median ΔP(AI), classification "
            "flip rate, flips_to_ai vs flips_to_real, plus the run's cohort size.\n"
            "- `accuracy_and_flip_by_transformation.png` — bar chart with accuracy, "
            "F1 and classification flip rate; overlay of mean ΔP(AI).\n"
            "- `delta_p_ai_boxplot.png` — per-image ΔP(AI) distribution per "
            "non-control transformation.\n\n"
            "## Cohort\n\n"
            "Balanced 50/50 real/AI images drawn from "
            f"`{args.split_dir}` (in-distribution combined_dataset test split). "
            "Sample size and seed are recorded in `summary_results.json → run`.\n\n"
            "## Reproduction\n\n"
            "```bash\n"
            "python -m src.genai_detection.evaluation.visual_classifier_robustness \\\n"
            f"  --sample-size {args.sample_size} --seed {args.seed}\n"
            "```\n"
        ),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"  rows      : {len(rows)}")
    print(f"  csv       : {args.output_dir / 'detailed_results.csv'}")
    print(f"  summary   : {args.output_dir / 'summary_results.json'}")
    print(f"  figures   : accuracy_and_flip_by_transformation.png, delta_p_ai_boxplot.png")
    print(f"  wall clock: {summary_run['wall_clock_seconds']:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
