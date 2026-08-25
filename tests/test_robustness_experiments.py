"""
Unit tests for the robustness-experiment harness
(``src/genai_detection/evaluation/``).

The tests cover:

* Every transformation is deterministic and produces a valid image (or
  the byte-preserving control's exact bytes).
* Metadata stripping actually drops the ``Make``/``Model`` EXIF tags
  that were present on the source.
* The byte-preserving control retains its source SHA-256; a JPEG
  re-encode changes it.
* Metric helpers compute the rates the runner promises on small
  hand-built rows.
* The end-to-end ``--smoke`` invocation runs, writes every promised
  output file, and cleans up every temporary derivative directory.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src import PROJECT_ROOT
from src.genai_detection.evaluation import (
    RESULT_COLUMNS,
    SMOKE_TRANSFORMATIONS,
    TRANSFORMATIONS,
    classify_outcome,
    row_coverage,
    summarise,
)
from src.genai_detection.evaluation import robustness_experiments as runner
from src.genai_detection.hash_module import sha256_bytes


# ---------------------------------------------------------------------------
# Fixtures — tiny synthetic images that don't need any external download
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_jpeg_bytes() -> bytes:
    """A tiny colourful JPEG with a Make/Model EXIF block.

    Kept small (32×24) so the tests are fast and don't stress Pillow's
    JPEG encoder. The EXIF block is written via Pillow's built-in
    ``exif`` argument so the test doesn't depend on ExifTool at
    generation time.
    """
    img = Image.new("RGB", (32, 24), color=(60, 120, 200))
    # Add a simple gradient so re-encoding actually shifts bytes.
    for x in range(32):
        for y in range(24):
            img.putpixel((x, y), ((x * 8) % 255, (y * 10) % 255, ((x + y) * 5) % 255))
    exif = Image.Exif()
    exif[0x010F] = "TestMake"       # Make
    exif[0x0110] = "TestModel"      # Model
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, exif=exif.tobytes())
    return buf.getvalue()


@pytest.fixture(scope="module")
def synthetic_png_bytes() -> bytes:
    img = Image.new("RGB", (40, 30))
    for x in range(40):
        for y in range(30):
            img.putpixel((x, y), ((x + y) % 255, (x * 2) % 255, (y * 3) % 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Transformation contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(TRANSFORMATIONS))
def test_transformation_is_deterministic(name, synthetic_jpeg_bytes):
    """Running any transformation twice with the same input must produce
    the same bytes — otherwise the CSV becomes irreproducible."""
    tf = TRANSFORMATIONS[name]
    out1, sfx1 = tf.apply(synthetic_jpeg_bytes, ".jpg")
    out2, sfx2 = tf.apply(synthetic_jpeg_bytes, ".jpg")
    assert out1 == out2
    assert sfx1 == sfx2


@pytest.mark.parametrize("name", list(TRANSFORMATIONS))
def test_transformation_produces_decodable_image(name, synthetic_png_bytes):
    """Every transformation output must be a valid image Pillow can
    decode — the runner passes it to detectors that assume so."""
    tf = TRANSFORMATIONS[name]
    out, _ = tf.apply(synthetic_png_bytes, ".png")
    Image.open(io.BytesIO(out)).verify()


def test_original_copy_preserves_bytes_and_hash(synthetic_jpeg_bytes):
    """The control MUST be byte-identical — that's the whole point."""
    tf = TRANSFORMATIONS["original_copy"]
    out, sfx = tf.apply(synthetic_jpeg_bytes, ".jpg")
    assert out == synthetic_jpeg_bytes
    assert sfx == ".jpg"
    assert sha256_bytes(out) == sha256_bytes(synthetic_jpeg_bytes)


def test_jpeg_reencode_changes_hash(synthetic_jpeg_bytes):
    """Re-encoding at a different quality MUST change the digest, or the
    SHA-256 signal cannot be said to be measuring what we claim."""
    tf = TRANSFORMATIONS["jpeg_q70"]
    out, _ = tf.apply(synthetic_jpeg_bytes, ".jpg")
    assert sha256_bytes(out) != sha256_bytes(synthetic_jpeg_bytes)


def test_metadata_stripped_removes_exif_tags(synthetic_jpeg_bytes, tmp_path):
    """The metadata-stripped output must not carry the Make/Model tags
    that the source had. Uses exiftool when available (the pipeline's
    canonical reader) with a fallback to PIL's own EXIF parser."""
    tf = TRANSFORMATIONS["metadata_stripped"]
    out, sfx = tf.apply(synthetic_jpeg_bytes, ".jpg")
    stripped_path = tmp_path / f"stripped{sfx}"
    stripped_path.write_bytes(out)

    exiftool = shutil.which("exiftool")
    if exiftool is None:
        img = Image.open(stripped_path)
        exif = img.getexif()
        assert exif.get(0x010F) in (None, "")
        assert exif.get(0x0110) in (None, "")
        return

    result = subprocess.run(
        [exiftool, "-j", "-Make", "-Model", str(stripped_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout or "[{}]")[0]
    assert "Make" not in parsed
    assert "Model" not in parsed


def test_screenshot_rasterise_matches_padding(synthetic_png_bytes):
    tf = TRANSFORMATIONS["screenshot_rasterise"]
    out, _ = tf.apply(synthetic_png_bytes, ".png")
    img = Image.open(io.BytesIO(out))
    # Original 40x30 with 8-pixel padding on each side.
    assert img.size == (56, 46)


def test_crop_reduces_dimensions(synthetic_png_bytes):
    tf = TRANSFORMATIONS["crop_10pct"]
    out, _ = tf.apply(synthetic_png_bytes, ".png")
    img = Image.open(io.BytesIO(out))
    assert img.size[0] < 40 and img.size[1] < 30


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _row(**kwargs):
    """Build a plausible-shaped row for the metric tests. Missing
    columns default to empty string so the shape matches
    :data:`RESULT_COLUMNS`."""
    base = {c: "" for c in RESULT_COLUMNS}
    base.update(kwargs)
    return base


def test_classify_outcome_correct_incorrect_inconclusive():
    assert classify_outcome("exact_match", "exact_match") == "correct"
    assert classify_outcome("no_match", "exact_match") == "incorrect"
    assert classify_outcome("detector_unavailable", "detected") == "inconclusive"
    assert classify_outcome("not_detected", None) == "inconclusive"


def test_row_coverage_treats_unavailable_as_uncovered():
    assert row_coverage(_row(signal="c2pa", status="valid", detector_available=True))
    assert not row_coverage(
        _row(signal="c2pa", status="validator_unavailable", detector_available=False)
    )
    assert not row_coverage(_row(signal="trustmark", status="error", detector_available=True))


def test_summarise_computes_expected_rates():
    """Small hand-built rows: two positives detected on control, one
    positive missed after resize; one control never triggers."""
    rows = [
        _row(signal="sha256", transformation="original_copy",
             status="exact_match", sha256_exact_match=True,
             outcome="correct", detector_available=True, runtime_seconds=0.001),
        _row(signal="sha256", transformation="jpeg_q70",
             status="no_match", sha256_exact_match=False,
             outcome="correct", detector_available=True, runtime_seconds=0.002),
        _row(signal="trustmark", transformation="original_copy",
             cohort="watermarked_positive", status="detected",
             outcome="correct", detector_available=True, runtime_seconds=0.5),
        _row(signal="trustmark", transformation="resize_50",
             cohort="watermarked_positive", status="not_detected",
             outcome="inconclusive", detector_available=True, runtime_seconds=0.6),
        _row(signal="trustmark", transformation="original_copy",
             cohort="unwatermarked_control", status="not_detected",
             outcome="correct", detector_available=True, runtime_seconds=0.4),
        _row(signal="c2pa", transformation="original_copy",
             status="absent", expected_status="absent",
             outcome="correct", detector_available=True, runtime_seconds=0.05),
    ]
    summary = summarise(rows)
    assert summary["row_count"] == 6
    assert summary["sha256_exact_match_by_transformation"]["original_copy"]["rate"] == 1.0
    assert summary["sha256_exact_match_by_transformation"]["jpeg_q70"]["rate"] == 0.0
    tm = summary["trustmark_by_transformation"]
    assert tm["original_copy"]["true_positive_rate"] == 1.0
    assert tm["resize_50"]["true_positive_rate"] == 0.0
    assert tm["original_copy"]["false_positive_rate"] == 0.0
    coverage = summary["coverage_by_signal"]
    assert coverage["sha256"]["rate"] == 1.0
    assert coverage["c2pa"]["rate"] == 1.0
    # Runtime aggregation: check the mean/median exist and are numeric.
    for signal, stats in summary["runtime_by_signal"].items():
        assert stats["mean_seconds"] is not None
        assert stats["median_seconds"] is not None


# ---------------------------------------------------------------------------
# End-to-end smoke run
# ---------------------------------------------------------------------------


def _copy_first_n_samples(dst: Path, n: int) -> None:
    """Copy the first ``n`` alphabetically-named files from the repo's
    sample-image directory into ``dst``. Skipped-if-missing to keep the
    test suite green on a fresh clone that hasn't pulled the samples."""
    src_dir = PROJECT_ROOT / "data" / "sample_images"
    if not src_dir.exists():
        pytest.skip("data/sample_images/ not present")
    for p in sorted(src_dir.iterdir())[:n]:
        shutil.copy2(p, dst / p.name)


def test_smoke_run_writes_expected_outputs(tmp_path, monkeypatch):
    """End-to-end: --smoke over a two-image copy of the sample set must
    produce every promised output and leave no derivative directory
    behind. The test does not depend on TrustMark's weights being
    present — if the library or model is missing the runner reports the
    detector as unavailable, which is one of the outcomes we WANT to
    test."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _copy_first_n_samples(input_dir, 2)
    output_dir = tmp_path / "out"

    # Track every tempfile.mkdtemp call so we can assert cleanup below.
    created_dirs: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        p = real_mkdtemp(*args, **kwargs)
        created_dirs.append(p)
        return p

    monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)

    exit_code = runner.main([
        "--smoke",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--max-trustmark-positives", "0",  # keep the test fast; unavailable is fine
    ])
    assert exit_code == 0

    for name in (
        "experiment_config.json",
        "detailed_results.csv",
        "summary_results.json",
    ):
        assert (output_dir / name).exists(), f"missing output {name}"

    # Figures may be absent if matplotlib fails; if present they must be non-empty.
    for png in ("robustness_by_transformation.png", "signal_coverage.png"):
        p = output_dir / png
        if p.exists():
            assert p.stat().st_size > 0

    # Every derivative / registry tempdir made by the runner must be gone.
    for p in created_dirs:
        assert not Path(p).exists(), f"leftover temp dir: {p}"

    # Config sanity: transformations recorded verbatim, the smoke set was used.
    config = json.loads((output_dir / "experiment_config.json").read_text())
    assert config["smoke"] is True
    assert [t["name"] for t in config["transformations"]] == list(SMOKE_TRANSFORMATIONS)

    # Summary shape sanity — at minimum every signal must appear in coverage.
    summary = json.loads((output_dir / "summary_results.json").read_text())["summary"]
    assert set(summary["coverage_by_signal"]) >= {"sha256", "c2pa", "trustmark"}
