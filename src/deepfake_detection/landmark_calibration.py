"""
Landmark-retrieval threshold calibration experiment.

Closes the "landmark retrieval threshold uncalibrated" P1 gap identified
in METHODOLOGY_EVALUATION_GUIDE.md §14 (also GAPS.md #10).

What this does
--------------

1. Loads the committed FAISS landmark index
   (``src/deepfake_detection/models/landmarks_index.faiss`` +
   ``landmarks_metadata.json``, built from
   ``zguo0525/google-landmarks-v2-mini``).

2. Builds two cohorts:

   * **known-landmark cohort** — a small subset of the same
     ``zguo0525/google-landmarks-v2-mini`` split used to build the index,
     drawn strided so it covers many distinct landmark classes. These
     images ARE represented in the FAISS index (they were embedded when
     the index was built), so a similarity match against them tests the
     index's own retrieval behaviour under the DINOv2 embedding.
   * **non-landmark cohort** — real photos from
     ``data/visual/combined_dataset/test``. This cohort deliberately
     mixes indoor scenes, portraits, close-ups and generic outdoor
     photos — the images the FAISS index has no reason to match at high
     similarity. A small residual overlap (a real photo that happens to
     depict a Google Landmarks class) is possible and is reported.

3. Runs ``LandmarkIndex.search(image, top_k=10)`` on every cohort image
   and records the winning label's mean top-k cosine similarity.

4. Reports:

   * distribution statistics (min / 25 / 50 / 75 / max, mean) per cohort;
   * confusion counts and TP-rate / FP-rate at a grid of candidate
     thresholds (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70);
   * ROC-AUC of the "is-landmark" binary discrimination task;
   * Youden's-J optimum threshold on the cohort;
   * a distribution plot with both cohorts overlaid;
   * a threshold-performance table.

5. Evaluates the current **0.5** threshold explicitly — reports its
   TP-rate and FP-rate — and states whether it remains defensible on
   this cohort, whether another threshold is empirically better, or
   whether the distributions overlap too strongly for any reliable
   threshold.

Writes to a NEW, versioned output directory
(``outputs/landmark_calibration/``) — never touches
``src/deepfake_detection/models/`` or the existing landmark index.

The 0.5 threshold is NOT changed in code by this experiment. Any change
must be reported to David and applied through
``src/deepfake_detection/deepfake_classifier.py`` deliberately, not as a
side-effect of running this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src import PROJECT_ROOT
from src.deepfake_detection.deepfake_classifier import LandmarkIndex


DEFAULT_INDEX_PATH = PROJECT_ROOT / "src" / "deepfake_detection" / "models" / "landmarks_index.faiss"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "src" / "deepfake_detection" / "models" / "landmarks_metadata.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "landmark_calibration"
DEFAULT_COMBINED_TEST = PROJECT_ROOT / "data" / "visual" / "combined_dataset" / "test"

CURRENT_THRESHOLD = 0.5  # documented in DeepfakeClassifier.predict / predict_landmark

CANDIDATE_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


@dataclass
class Score:
    """One score row."""

    cohort: str  # "known_landmark" | "non_landmark"
    sample_id: str
    winning_label: str
    mean_cosine: float
    top_k: int


# ---------------------------------------------------------------------------
# Cohort assembly
# ---------------------------------------------------------------------------


def _load_known_landmark_cohort(
    n: int, split: str = "test",
) -> list[tuple[str, "Image.Image"]]:
    """Take ``n`` images from ``zguo0525/google-landmarks-v2-mini``.

    Downloaded through HuggingFace ``datasets`` and cached locally.

    ``split="test"`` (default, disjoint from index) — the 6 206-row test
    split whose images were **not** embedded when the FAISS index was
    built (``initialise_index.py`` uses ``split="train"`` only). Every
    test-split image belongs to a class the index knows, so retrieval
    can match a class without matching the query image itself. This is
    the genuinely held-out condition — use for methodologically valid
    calibration.

    ``split="train"`` — the same split ``initialise_index.py`` embedded.
    Retained ONLY so the original leaked-cohort result can be reproduced
    for comparison. Do not use for a claim about the retrieval's
    generalisation to unseen images.
    """
    from datasets import load_dataset

    ds = load_dataset("zguo0525/google-landmarks-v2-mini", split=split)
    total = len(ds)
    if n > total:
        n = total
    step = total / n
    picked = [int(round(i * step)) for i in range(n)]
    out: list[tuple[str, "Image.Image"]] = []
    class_names = ds.features["label"].names
    for idx in picked:
        row = ds[idx]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        out.append((f"gl_{split}__{idx:05d}__cls_{class_names[row['label']]}", img))
    return out


def _load_non_landmark_cohort(n: int, split_dir: Path) -> list[tuple[str, "Image.Image"]]:
    """Sample ``n`` real photos from ``combined_dataset/test``.

    Reals only (label==0) — AI images are not appropriate as a
    "non-landmark natural scene" cohort. Strided by streaming index so
    the sample is not front-loaded with the first shard.
    """
    from src.genai_detection.visual_module.evaluation import iter_split_images

    reals: list[int] = []
    for idx, (_img, label) in enumerate(iter_split_images(str(split_dir))):
        if label == 0:
            reals.append(idx)
    if not reals:
        raise RuntimeError(f"no real images in {split_dir}")
    if n > len(reals):
        n = len(reals)
    step = len(reals) / n
    picked = {reals[int(round(i * step))] for i in range(n)}

    out: list[tuple[str, "Image.Image"]] = []
    for idx, (img, label) in enumerate(iter_split_images(str(split_dir))):
        if idx not in picked or label != 0:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        out.append((f"comb__{idx:05d}", img))
        if len(out) == len(picked):
            break
    return out


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def _describe(cosines: list[float]) -> dict[str, float]:
    if not cosines:
        return {"n": 0}
    a = np.asarray(cosines, dtype=float)
    return {
        "n": len(a),
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
        "std": float(a.std(ddof=0)),
    }


def _threshold_metrics(
    known: list[float], non_landmark: list[float], threshold: float
) -> dict[str, Any]:
    tp = sum(1 for c in known if c >= threshold)
    fn = sum(1 for c in known if c < threshold)
    fp = sum(1 for c in non_landmark if c >= threshold)
    tn = sum(1 for c in non_landmark if c < threshold)
    n_pos = tp + fn
    n_neg = fp + tn
    tpr = tp / n_pos if n_pos else None
    fpr = fp / n_neg if n_neg else None
    return {
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tpr,
        "fpr": fpr,
        "youden_j": (tpr - fpr) if (tpr is not None and fpr is not None) else None,
    }


def _auc(known: list[float], non_landmark: list[float]) -> float | None:
    if not known or not non_landmark:
        return None
    from sklearn.metrics import roc_auc_score

    y = [1] * len(known) + [0] * len(non_landmark)
    scores = list(known) + list(non_landmark)
    return float(roc_auc_score(y, scores))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_distributions(
    known: list[float], non_landmark: list[float], threshold: float, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(non_landmark, bins=bins, alpha=0.65, label=f"Non-landmark (n={len(non_landmark)})", color="#4c72b0")
    ax.hist(known, bins=bins, alpha=0.65, label=f"Known landmark (n={len(known)})", color="#dd8452")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.0, label=f"Current threshold ({threshold})")
    ax.set_xlabel("Mean top-k cosine similarity of winning landmark label")
    ax.set_ylabel("Count")
    ax.set_title("Landmark-retrieval score distributions by cohort")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc(known: list[float], non_landmark: list[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    y = [1] * len(known) + [0] * len(non_landmark)
    scores = list(known) + list(non_landmark)
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"Known-landmark vs non-landmark (AUC {auc:.3f})", color="#c44e52")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Landmark-retrieval ROC")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.deepfake_detection.landmark_calibration"
    )
    p.add_argument("--n-known", type=int, default=100,
                   help="Number of known-landmark images to score.")
    p.add_argument("--n-non-landmark", type=int, default=100,
                   help="Number of non-landmark real images to score.")
    p.add_argument("--known-split", type=str, default="test",
                   choices=["train", "test"],
                   help="Landmark dataset split to draw the known-landmark "
                        "cohort from. 'test' (default) is disjoint from the "
                        "FAISS index (which was built from 'train'). Use "
                        "'train' only to reproduce the historical leaked-cohort "
                        "result.")
    p.add_argument("--split-dir", type=Path, default=DEFAULT_COMBINED_TEST)
    p.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    p.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading landmark index from {args.index_path} ...")
    lm = LandmarkIndex(
        index_path=str(args.index_path),
        metadata_path=str(args.metadata_path),
    )

    print(f"Loading known-landmark cohort ({args.n_known}) from split={args.known_split!r} ...")
    known_imgs = _load_known_landmark_cohort(args.n_known, split=args.known_split)

    print(f"Loading non-landmark cohort ({args.n_non_landmark}) ...")
    non_imgs = _load_non_landmark_cohort(args.n_non_landmark, args.split_dir)

    scores: list[Score] = []
    for cohort_name, cohort in [
        ("known_landmark", known_imgs),
        ("non_landmark", non_imgs),
    ]:
        print(f"Scoring {cohort_name} cohort (n={len(cohort)}) ...")
        t0 = time.perf_counter()
        for sid, img in cohort:
            res = lm.search(img, top_k=args.top_k, similarity_threshold=0.0)
            # We deliberately pass threshold=0.0 so `search` never returns
            # the "Unknown" fallback and always names the winning label
            # with its mean cosine — this experiment IS the threshold
            # study, so we want the raw score every time.
            label = res.get("label", "N/A")
            cosine = float(res.get("confidence", 0.0))
            scores.append(
                Score(
                    cohort=cohort_name,
                    sample_id=sid,
                    winning_label=label,
                    mean_cosine=cosine,
                    top_k=args.top_k,
                )
            )
        print(f"  scored {len(cohort)} in {time.perf_counter() - t0:.1f}s")

    known_cos = [s.mean_cosine for s in scores if s.cohort == "known_landmark"]
    non_cos = [s.mean_cosine for s in scores if s.cohort == "non_landmark"]

    # Threshold sweep
    threshold_rows = [
        _threshold_metrics(known_cos, non_cos, t) for t in CANDIDATE_THRESHOLDS
    ]

    # Youden's J optimum from a finer sweep (0.01 step)
    fine = np.linspace(0.10, 0.90, 81)
    best = max(
        (_threshold_metrics(known_cos, non_cos, float(t)) for t in fine),
        key=lambda r: (r["youden_j"] if r["youden_j"] is not None else -1),
    )

    auc = _auc(known_cos, non_cos)

    current = _threshold_metrics(known_cos, non_cos, CURRENT_THRESHOLD)

    # Interpretation logic — deterministic, prints a clear verdict.
    def _interpret(auc_val, current_row, best_row):
        if auc_val is None:
            return "insufficient data to assess separability"
        if auc_val < 0.65:
            return (
                "Distributions overlap heavily (AUC < 0.65). No single "
                "cosine threshold provides reliable separation between "
                "known-landmark and non-landmark natural images on this "
                "cohort. The current 0.5 threshold should be reported as "
                "a heuristic gate, not a calibrated classifier."
            )
        diff = best_row["youden_j"] - current_row["youden_j"]
        if diff <= 0.05:
            return (
                f"The current 0.5 threshold remains defensible "
                f"(Youden's J {current_row['youden_j']:.3f}); the best "
                f"threshold on this cohort ({best_row['threshold']:.2f}) "
                f"improves Youden's J by only {diff:+.3f}, which is not "
                f"a clearly better operating point."
            )
        return (
            f"A different threshold ({best_row['threshold']:.2f}) is "
            f"empirically better on this cohort (Youden's J "
            f"{best_row['youden_j']:.3f} vs {current_row['youden_j']:.3f} "
            f"at the current 0.5). Reporting the change to David is "
            f"required before it is applied in code."
        )

    interpretation = _interpret(auc, current, best)

    # ---------------- Write outputs ----------------
    # Detailed CSV
    with (out / "detailed_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["cohort", "sample_id", "winning_label", "mean_cosine", "top_k"],
        )
        w.writeheader()
        for s in scores:
            w.writerow(
                {
                    "cohort": s.cohort,
                    "sample_id": s.sample_id,
                    "winning_label": s.winning_label,
                    "mean_cosine": f"{s.mean_cosine:.6f}",
                    "top_k": s.top_k,
                }
            )

    summary = {
        "cohorts": {
            "known_landmark": {"n": len(known_cos), **_describe(known_cos)},
            "non_landmark": {"n": len(non_cos), **_describe(non_cos)},
        },
        "auc_known_vs_non_landmark": auc,
        "current_threshold": {
            "value": CURRENT_THRESHOLD,
            **current,
        },
        "best_threshold_youden_j": best,
        "threshold_grid": threshold_rows,
        "interpretation": interpretation,
        "run": {
            "index_path": str(args.index_path),
            "metadata_path": str(args.metadata_path),
            "split_dir": str(args.split_dir),
            "top_k": args.top_k,
            "candidate_thresholds": CANDIDATE_THRESHOLDS,
            "known_landmark_split": args.known_split,
            "known_landmark_disjoint_from_index": (args.known_split == "test"),
        },
    }
    (out / "summary_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Threshold sweep table (CSV)
    with (out / "threshold_grid.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["threshold", "tp", "fn", "fp", "tn", "tpr", "fpr", "youden_j"]
        )
        w.writeheader()
        for row in threshold_rows:
            w.writerow(row)

    # Plots
    try:
        _plot_distributions(known_cos, non_cos, CURRENT_THRESHOLD, out / "score_distributions.png")
        _plot_roc(known_cos, non_cos, out / "roc_landmark_vs_non_landmark.png")
    except Exception as exc:  # pragma: no cover
        print(f"  [plot] failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    (out / "README.md").write_text(
        (
            "# Landmark-retrieval threshold calibration — outputs\n\n"
            "Regenerated on every run of "
            "`python -m src.deepfake_detection.landmark_calibration`.\n\n"
            "## Files\n\n"
            "- `detailed_results.csv` — one row per (cohort, sample, "
            "winning_label, mean_cosine, top_k).\n"
            "- `summary_results.json` — distribution statistics per "
            "cohort, ROC-AUC, threshold grid, Youden's-J optimum, "
            "current-threshold report, plain-English interpretation.\n"
            "- `threshold_grid.csv` — TPR/FPR/Youden's J at each of "
            f"{CANDIDATE_THRESHOLDS}.\n"
            "- `score_distributions.png` — overlaid histograms of the "
            "winning-label mean cosine per cohort, with the current "
            f"threshold ({CURRENT_THRESHOLD}) marked.\n"
            "- `roc_landmark_vs_non_landmark.png` — ROC of the "
            "\"is-landmark\" binary discrimination task on this cohort.\n\n"
            "## Cohorts\n\n"
            "- **known_landmark** — subset of "
            "`zguo0525/google-landmarks-v2-mini` (the same dataset the "
            "committed FAISS index was built from). These images are in "
            "the index, so this cohort measures the index's ability to "
            "match a query drawn from the underlying reference "
            "distribution — an upper bound on retrieval accuracy.\n"
            "- **non_landmark** — real photos from "
            "`data/visual/combined_dataset/test`. Small residual "
            "landmark-adjacency is possible (a real photo may happen to "
            "depict a Google-Landmarks class) and is reported as a "
            "false-positive count in the threshold table.\n\n"
            "## What this experiment does NOT do\n\n"
            "- Does not modify the 0.5 cosine threshold in "
            "`src/deepfake_detection/deepfake_classifier.py`. Any "
            "change is a deliberate decision for David and must be "
            "reported in the write-up.\n"
            "- Does not claim that the landmark-retrieval score is a "
            "calibrated probability. It is a mean top-k cosine.\n"
            "- Does not evaluate whether a matched landmark image is "
            "AI-generated. That is a separate question answered "
            "elsewhere in the pipeline.\n"
        ),
        encoding="utf-8",
    )

    print()
    print("Distribution statistics:")
    print(f"  known landmark: {summary['cohorts']['known_landmark']}")
    print(f"  non-landmark:  {summary['cohorts']['non_landmark']}")
    print()
    print(f"AUC (known vs non-landmark): {auc}")
    print(f"Current threshold {CURRENT_THRESHOLD}: TPR={current['tpr']}, FPR={current['fpr']}, J={current['youden_j']}")
    print(f"Youden's-J optimum: threshold={best['threshold']:.3f}, TPR={best['tpr']}, FPR={best['fpr']}, J={best['youden_j']:.3f}")
    print()
    print("Interpretation:")
    print("  " + interpretation)
    print()
    print(f"CSV : {out / 'detailed_results.csv'}")
    print(f"JSON: {out / 'summary_results.json'}")
    print(f"FIG : {out / 'score_distributions.png'}, {out / 'roc_landmark_vs_non_landmark.png'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
