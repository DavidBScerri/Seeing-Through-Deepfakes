"""
Aggregation helpers for the robustness experiment.

Keeps arithmetic separate from the runner and the plots so the tests can
pin the metric semantics on tiny hand-built rows without needing
detectors, temp files, or plotting backends. The public shapes are
intentionally boring — dicts of floats keyed by transformation name /
signal — so the JSON summary is trivially inspectable.

Terminology (deliberately narrow, mirrors the module contracts):

* **Coverage** — a signal is *covered* on a row iff its detector was
  technically able to evaluate the input (i.e. status is neither
  UNAVAILABLE nor an implementation ERROR / INVALID_REGISTRY /
  DETECTOR_UNAVAILABLE / VALIDATOR_UNAVAILABLE). A no-match / not-
  detected / absent result is covered.
* **Survival** — for the C2PA and TrustMark signals only, the fraction
  of covered rows in the "signal-should-survive" cohort whose status
  still reads VALID / DETECTED after the transformation. For SHA-256
  survival is the exact-match rate of the byte-preserving control.
* **True/false positive rate** (TrustMark only) — TP-rate over the
  watermarked-positive cohort, FP-rate over the unwatermarked-control
  cohort. Neither number is meaningful for the other signals — SHA-256
  is byte-exact by construction and C2PA validation is not a binary
  classifier.

Nothing here reads back into a single "accuracy" number: the three
signals answer different questions and must be reported separately.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable


# Rows carry raw string status values from each detector. Keeping the
# names symbolic here means the metric layer never has to import the
# module enums — the runner is the single place that translates enum
# values to strings.

# ---------------------------------------------------------------------------
# Row schema (documentation only; the runner emits plain dicts)
# ---------------------------------------------------------------------------

RESULT_COLUMNS: tuple[str, ...] = (
    "image_id",
    "source_path",
    "cohort",
    "transformation",
    "params_json",
    "signal",
    "detector_available",
    "status",
    "expected_status",
    "outcome",
    "runtime_seconds",
    "error_details",
    "trustmark_variant",
    "trustmark_schema_version",
    "c2pa_origin_claim",
    "sha256_source",
    "sha256_transformed",
    "sha256_exact_match",
)


# Statuses that mean "the detector was not in a position to answer".
# Kept here so both the metrics helpers and the runner agree on the
# boundary between coverage and outcome.
UNAVAILABLE_STATUSES: frozenset[str] = frozenset({
    "detector_unavailable",
    "validator_unavailable",
    "registry_unavailable",
    "invalid_registry",
    "unsupported",
    "unsupported_format",
})

ERROR_STATUSES: frozenset[str] = frozenset({"error"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    return float(mean(vals)) if vals else None


def _median_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    return float(median(vals)) if vals else None


def _rate(numerator: int, denominator: int) -> float | None:
    """Return ``numerator / denominator`` as a float, or ``None`` when the
    denominator is zero. ``None`` beats ``0.0`` here — an empty cohort
    means "we can't say", not "it fails every time"."""
    if denominator == 0:
        return None
    return numerator / denominator


# ---------------------------------------------------------------------------
# Row-level classification
# ---------------------------------------------------------------------------


def row_coverage(row: dict) -> bool:
    """True iff the detector on this row was technically able to answer.

    A row with an UNAVAILABLE / ERROR-family status is NOT covered — the
    detector could not run or crashed. A NO_MATCH / NOT_DETECTED /
    ABSENT row IS covered — the detector answered, and the answer was
    negative.
    """
    if not row.get("detector_available", True):
        return False
    status = str(row.get("status", "")).lower()
    if status in UNAVAILABLE_STATUSES or status in ERROR_STATUSES:
        return False
    return True


def _classify_outcome(status: str, expected: str | None) -> str:
    """Compare ``status`` against ``expected`` and return one of
    ``correct`` / ``incorrect`` / ``inconclusive``.

    ``inconclusive`` covers three cases: the expected value is ``None``
    (no ground truth to compare against — e.g. the WM survival on a
    non-control transformation, where the honest answer is "we're
    measuring, not judging"), the detector was unavailable, or the
    detector raised an error.
    """
    s = (status or "").lower()
    if s in UNAVAILABLE_STATUSES or s in ERROR_STATUSES:
        return "inconclusive"
    if expected is None:
        return "inconclusive"
    return "correct" if s == expected.lower() else "incorrect"


def classify_outcome(status: str, expected: str | None) -> str:
    """Public alias — kept so the runner and tests share one entry point."""
    return _classify_outcome(status, expected)


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def summarise_runtimes(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    """Mean / median wall-clock time per signal.

    Runtime is a per-signal property — the SHA-256 hash is orders of
    magnitude cheaper than a TrustMark decode — so mixing them into one
    "average detector time" is misleading.
    """
    per_signal: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        # We time every row, including unavailable ones (they still cost
        # the caller some setup time). Non-numeric values are ignored so
        # a NaN never poisons the summary.
        rt = row.get("runtime_seconds")
        if isinstance(rt, (int, float)):
            per_signal[row["signal"]].append(float(rt))
    return {
        signal: {
            "mean_seconds": _mean_or_none(values),
            "median_seconds": _median_or_none(values),
            "count": len(values),
        }
        for signal, values in per_signal.items()
    }


def coverage_by_signal(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """Coverage rate per signal — fraction of rows where the detector
    actually answered.

    Reports raw counts alongside the rate so a reader can see whether a
    100% coverage came from ten rows or a thousand.
    """
    per_signal_total: dict[str, int] = defaultdict(int)
    per_signal_covered: dict[str, int] = defaultdict(int)
    for row in rows:
        signal = row["signal"]
        per_signal_total[signal] += 1
        if row_coverage(row):
            per_signal_covered[signal] += 1
    return {
        signal: {
            "covered": per_signal_covered[signal],
            "total": per_signal_total[signal],
            "rate": _rate(per_signal_covered[signal], per_signal_total[signal]),
        }
        for signal in per_signal_total
    }


def sha256_exact_match_rate(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """SHA-256 exact-match rate per transformation.

    Only SHA-256 rows count. The expected result on ``original_copy`` is
    100%; on every other transformation it should collapse to 0% —
    that's the byte-exact contract we're demonstrating.
    """
    per_tf_total: dict[str, int] = defaultdict(int)
    per_tf_match: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["signal"] != "sha256":
            continue
        tf = row["transformation"]
        per_tf_total[tf] += 1
        if row.get("sha256_exact_match"):
            per_tf_match[tf] += 1
    return {
        tf: {
            "matches": per_tf_match[tf],
            "total": per_tf_total[tf],
            "rate": _rate(per_tf_match[tf], per_tf_total[tf]),
        }
        for tf in per_tf_total
    }


def c2pa_status_breakdown(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """C2PA status breakdown per transformation.

    Emits raw counts for VALID / ABSENT / INVALID_OR_TAMPERED /
    UNTRUSTED_SIGNER / (validator) unavailable / other. The rates are
    normalised over rows for that transformation, and the "survival"
    figure is the VALID rate restricted to images whose ORIGINAL
    validated — the only cohort for which a survival number is
    meaningful.
    """
    per_tf: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    survival_total: dict[str, int] = defaultdict(int)
    survival_valid: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["signal"] != "c2pa":
            continue
        tf = row["transformation"]
        status = str(row.get("status", "")).lower()
        per_tf[tf][status] += 1
        per_tf[tf]["_total"] += 1
        # Survival cohort: only images whose expected_status on this
        # transformation is "valid" (which the runner sets iff the
        # original manifest validated). Otherwise a "valid" rate would
        # mix samples that never had a manifest with ones that did.
        if str(row.get("expected_status", "")).lower() == "valid":
            survival_total[tf] += 1
            if status == "valid":
                survival_valid[tf] += 1
    out: dict[str, dict[str, Any]] = {}
    for tf, counts in per_tf.items():
        total = counts.pop("_total")
        breakdown = {
            k: {"count": v, "rate": _rate(v, total)}
            for k, v in counts.items()
        }
        out[tf] = {
            "total": total,
            "breakdown": breakdown,
            "valid_survival_rate": _rate(survival_valid[tf], survival_total[tf]),
            "valid_survival_cohort_size": survival_total[tf],
        }
    return out


def trustmark_rates(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """TrustMark TP / FP / survival per transformation.

    * TP-rate = ``detected / total`` over rows whose ``cohort`` is
      ``watermarked_positive``.
    * FP-rate = ``detected / total`` over rows whose ``cohort`` is
      ``unwatermarked_control``.
    * Survival rate = TP-rate specifically on the watermarked cohort —
      spelled out separately so a plot of "did the watermark survive
      this transformation" doesn't have to know the cohort taxonomy.
    """
    tp_total: dict[str, int] = defaultdict(int)
    tp_detected: dict[str, int] = defaultdict(int)
    fp_total: dict[str, int] = defaultdict(int)
    fp_detected: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["signal"] != "trustmark":
            continue
        tf = row["transformation"]
        detected = str(row.get("status", "")).lower() == "detected"
        cohort = row.get("cohort")
        if cohort == "watermarked_positive":
            tp_total[tf] += 1
            if detected:
                tp_detected[tf] += 1
        elif cohort == "unwatermarked_control":
            fp_total[tf] += 1
            if detected:
                fp_detected[tf] += 1
    keys = set(tp_total) | set(fp_total)
    return {
        tf: {
            "true_positive_rate": _rate(tp_detected[tf], tp_total[tf]),
            "true_positive_count": tp_detected[tf],
            "positive_cohort_size": tp_total[tf],
            "false_positive_rate": _rate(fp_detected[tf], fp_total[tf]),
            "false_positive_count": fp_detected[tf],
            "negative_cohort_size": fp_total[tf],
            "survival_rate": _rate(tp_detected[tf], tp_total[tf]),
        }
        for tf in keys
    }


def inconclusive_rate(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """Fraction of ``inconclusive`` outcomes per signal.

    Split out because "the signal could not tell us anything" is a
    result in its own right — the write-up needs to be clear about how
    often each detector actually took a stance.
    """
    per_signal_total: dict[str, int] = defaultdict(int)
    per_signal_inconclusive: dict[str, int] = defaultdict(int)
    for row in rows:
        signal = row["signal"]
        per_signal_total[signal] += 1
        if row.get("outcome") == "inconclusive":
            per_signal_inconclusive[signal] += 1
    return {
        signal: {
            "inconclusive": per_signal_inconclusive[signal],
            "total": per_signal_total[signal],
            "rate": _rate(per_signal_inconclusive[signal], per_signal_total[signal]),
        }
        for signal in per_signal_total
    }


def summarise(rows: list[dict]) -> dict[str, Any]:
    """Compose every per-signal summary into one dict for
    ``summary_results.json``.

    The runner writes this verbatim; the plots read from it. Nothing
    here derives a global "accuracy" — the three signals answer
    different questions and are reported separately by design.
    """
    return {
        "row_count": len(rows),
        "coverage_by_signal": coverage_by_signal(rows),
        "inconclusive_rate": inconclusive_rate(rows),
        "runtime_by_signal": summarise_runtimes(rows),
        "sha256_exact_match_by_transformation": sha256_exact_match_rate(rows),
        "c2pa_by_transformation": c2pa_status_breakdown(rows),
        "trustmark_by_transformation": trustmark_rates(rows),
    }


__all__ = [
    "RESULT_COLUMNS",
    "UNAVAILABLE_STATUSES",
    "ERROR_STATUSES",
    "classify_outcome",
    "row_coverage",
    "summarise",
    "summarise_runtimes",
    "coverage_by_signal",
    "sha256_exact_match_rate",
    "c2pa_status_breakdown",
    "trustmark_rates",
    "inconclusive_rate",
]
