"""
Sample a balanced set of source images from ``combined_dataset/test`` and
write them to disk as PNG files.

Used to prepare a 25+ image source directory for the TrustMark P1 rerun
(so the existing ``robustness_experiments`` harness can be pointed at it
with ``--input-dir --max-trustmark-positives 25``). Also usable as a
generic "give me N balanced sources on disk" helper.

The images are re-encoded to PNG (lossless), keep their streaming index
as their filename prefix, and record their ground-truth label in the
filename per this project's convention (``{real|ai}_source_NN.png``).

    python -m src.genai_detection.evaluation.prepare_source_cohort \\
        --n 26 --output-dir data/watermark/run_02_cohort --seed 42
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image

from src import PROJECT_ROOT
from src.genai_detection.visual_module.evaluation import iter_split_images


DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "visual" / "combined_dataset" / "test"


def _strided(indices, k):
    step = len(indices) / k
    return [indices[int(round(i * step))] for i in range(k)]


def prepare(split_dir: Path, n: int, output_dir: Path, seed: int) -> list[Path]:
    per_class = n // 2
    reals: list[int] = []
    ais: list[int] = []
    for idx, (_img, label) in enumerate(iter_split_images(str(split_dir))):
        (reals if label == 0 else ais).append(idx)

    picked_real = _strided(reals, per_class)
    picked_ai = _strided(ais, n - per_class)
    picked_set = set(picked_real + picked_ai)
    label_by_index = {i: 0 for i in picked_real}
    label_by_index.update({i: 1 for i in picked_ai})

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for idx, (img, _label) in enumerate(iter_split_images(str(split_dir))):
        if idx not in picked_set:
            continue
        label = label_by_index[idx]
        prefix = "real" if label == 0 else "ai"
        # No metadata → filename says only what class the image is,
        # matching the sample-image convention `{real|ai}-metadata-place-face.png`.
        path = output_dir / f"{prefix}-metadata-place-face_src{idx:05d}.png"
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(path, format="PNG", optimize=False)
        saved.append(path)
        if len(saved) == n:
            break

    print(f"Wrote {len(saved)} images to {output_dir}")
    _ = seed
    return saved


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.genai_detection.evaluation.prepare_source_cohort")
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=26)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    prepare(args.split_dir, args.n, args.output_dir, args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
