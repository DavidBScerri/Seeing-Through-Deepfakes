"""
Deterministic image transformations used by the robustness experiment.

Each transformation is a small pure-ish function ``(src_bytes,
src_suffix, params) -> (dst_bytes, dst_suffix)``. Byte outputs — never
paths — so the caller controls storage: the experiment runner writes
them to a temporary directory, hashes them, feeds them to each signal,
and deletes them before exit.

Rules the transformations follow:

* Determinism. Nothing here consults ``random`` or wall-clock time.
  Re-running with the same input bytes and the same parameters must
  yield byte-identical output; the tests pin that invariant.
* ``original_copy`` is byte-identical to the source, keeping the file
  extension. It is the ONLY entry that a byte-exact SHA-256 must
  survive; everything else deliberately alters the pixels or the
  container.
* Every parameter that could change output is captured in the
  transformation's entry in :data:`TRANSFORMATIONS`, and the runner
  serialises the full spec into ``experiment_config.json`` so a reader
  can reproduce every derivative from the source images alone.
* Metadata stripping actually strips: the Pillow ``save`` calls
  intentionally do not pass ``exif=``/``xmp=``, so the resulting file
  carries no EXIF / XMP block. The unit tests check that the standard
  ``Make``/``Model`` tags disappear.

The set is deliberately small — the point is to characterise how each
signal behaves under a handful of illustrative operations, not to sweep
a vast grid of jpeg qualities.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image, ImageEnhance


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


def _open_rgb(src_bytes: bytes) -> Image.Image:
    """Decode ``src_bytes`` into an RGB Pillow image.

    RGB conversion is done here so downstream transformations don't have
    to worry about palette / alpha modes when re-encoding to JPEG. PNG
    outputs also normalise to RGB — the experiment measures signal
    survival under a re-encode, not under alpha/palette juggling.
    """
    img = Image.open(io.BytesIO(src_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _encode(img: Image.Image, fmt: str, **save_kwargs: Any) -> bytes:
    """Serialise ``img`` into memory, forcing ``fmt`` and dropping any
    metadata block that would otherwise be written."""
    buf = io.BytesIO()
    # Pillow will preserve embedded ICC / EXIF for some formats when it
    # can. We deliberately don't forward any of that here — the
    # transformation contract is "the pixels are the only thing
    # preserved unless the transformation says otherwise".
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Transformation implementations
# ---------------------------------------------------------------------------


def _identity(src_bytes: bytes, src_suffix: str, params: dict) -> tuple[bytes, str]:
    """Byte-preserving copy. The output equals the input, extension included."""
    return src_bytes, src_suffix


def _strip_metadata(src_bytes: bytes, src_suffix: str, params: dict) -> tuple[bytes, str]:
    """Re-encode the pixels in the source format with no EXIF/XMP block.

    Uses lossless PNG output when the source is PNG (metadata gone, pixels
    intact); uses a very high-quality JPEG for JPEG sources (pixels almost
    intact, metadata gone). The point is to isolate metadata removal from
    heavy re-encoding — those cases have their own dedicated entries.
    """
    img = _open_rgb(src_bytes)
    suffix = src_suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return _encode(img, "JPEG", quality=100, subsampling=0, optimize=False), ".jpg"
    return _encode(img, "PNG", optimize=False), ".png"


def _jpeg_reencode(src_bytes: bytes, src_suffix: str, params: dict) -> tuple[bytes, str]:
    quality = int(params.get("quality", 90))
    img = _open_rgb(src_bytes)
    return _encode(img, "JPEG", quality=quality, optimize=False), ".jpg"


def _resize(src_bytes: bytes, src_suffix: str, params: dict) -> tuple[bytes, str]:
    scale = float(params.get("scale", 0.75))
    img = _open_rgb(src_bytes)
    w, h = img.size
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    # Encode as high-quality JPEG so the change we're measuring is the
    # resize, not a heavy re-encode. Suffix ".jpg" so tests can spot the
    # container change explicitly.
    return _encode(resized, "JPEG", quality=95, optimize=False), ".jpg"


def _centre_crop(src_bytes: bytes, src_suffix: str, params: dict) -> tuple[bytes, str]:
    """Remove ``fraction`` of the total width and height from the borders.

    ``fraction=0.05`` removes ~5% total (2.5% from each side); the
    resulting image loses only its outer margin.
    """
    fraction = float(params.get("fraction", 0.05))
    img = _open_rgb(src_bytes)
    w, h = img.size
    dx = int(round(w * fraction / 2))
    dy = int(round(h * fraction / 2))
    box = (dx, dy, max(dx + 1, w - dx), max(dy + 1, h - dy))
    cropped = img.crop(box)
    return _encode(cropped, "JPEG", quality=95, optimize=False), ".jpg"


def _format_convert(src_bytes: bytes, src_suffix: str, params: dict) -> tuple[bytes, str]:
    """PNG↔JPEG conversion. Direction is chosen from the source suffix so
    one entry covers both directions naturally."""
    img = _open_rgb(src_bytes)
    if src_suffix.lower() == ".png":
        return _encode(img, "JPEG", quality=95, optimize=False), ".jpg"
    return _encode(img, "PNG", optimize=False), ".png"


def _screenshot_rasterise(src_bytes: bytes, src_suffix: str, params: dict) -> tuple[bytes, str]:
    """Approximate a screenshot: paste the image into a slightly larger
    neutral-grey canvas and re-encode as a lossless PNG.

    This is deliberately a mild proxy for "someone took a screenshot" —
    the real thing would also involve viewer chrome, subpixel rendering,
    and a colour-space round-trip. What matters here is that the file
    bytes and any embedded metadata are replaced wholesale while the
    pixels roughly survive.
    """
    padding = int(params.get("padding", 8))
    bg = tuple(params.get("background", (240, 240, 240)))
    img = _open_rgb(src_bytes)
    w, h = img.size
    canvas = Image.new("RGB", (w + 2 * padding, h + 2 * padding), bg)
    canvas.paste(img, (padding, padding))
    return _encode(canvas, "PNG", optimize=False), ".png"


def _brightness(src_bytes: bytes, src_suffix: str, params: dict) -> tuple[bytes, str]:
    factor = float(params.get("factor", 1.1))
    img = _open_rgb(src_bytes)
    adjusted = ImageEnhance.Brightness(img).enhance(factor)
    return _encode(adjusted, "JPEG", quality=95, optimize=False), ".jpg"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transformation:
    """One entry in the transformation registry.

    Kept a plain dataclass so it serialises cleanly into
    ``experiment_config.json`` — the runner dumps ``asdict(...)`` for
    each active transformation so a reader can rebuild every derivative
    from the sources alone.
    """

    name: str
    description: str
    fn: Callable[[bytes, str, dict], tuple[bytes, str]]
    params: dict[str, Any] = field(default_factory=dict)
    tag: str = "other"
    """Coarse category for the summary — ``control`` for the byte-preserving
    copy, ``metadata`` for the stripping entry, ``encoding`` /
    ``geometry`` / ``format`` / ``pixel`` for the others. Only used to
    order the summary chart."""

    def apply(self, src_bytes: bytes, src_suffix: str) -> tuple[bytes, str]:
        return self.fn(src_bytes, src_suffix, dict(self.params))


TRANSFORMATIONS: dict[str, Transformation] = {
    t.name: t for t in [
        Transformation(
            name="original_copy",
            description="Byte-preserving copy of the source (control).",
            fn=_identity,
            tag="control",
        ),
        Transformation(
            name="metadata_stripped",
            description="Re-encode in source format with no EXIF/XMP block.",
            fn=_strip_metadata,
            tag="metadata",
        ),
        Transformation(
            name="jpeg_q90",
            description="JPEG re-encode at quality 90.",
            fn=_jpeg_reencode,
            params={"quality": 90},
            tag="encoding",
        ),
        Transformation(
            name="jpeg_q70",
            description="JPEG re-encode at quality 70.",
            fn=_jpeg_reencode,
            params={"quality": 70},
            tag="encoding",
        ),
        Transformation(
            name="jpeg_q50",
            description="JPEG re-encode at quality 50.",
            fn=_jpeg_reencode,
            params={"quality": 50},
            tag="encoding",
        ),
        Transformation(
            name="resize_75",
            description="Lanczos resize to 75% of each dimension, JPEG-95 output.",
            fn=_resize,
            params={"scale": 0.75},
            tag="geometry",
        ),
        Transformation(
            name="resize_50",
            description="Lanczos resize to 50% of each dimension, JPEG-95 output.",
            fn=_resize,
            params={"scale": 0.50},
            tag="geometry",
        ),
        Transformation(
            name="crop_5pct",
            description="Centre crop removing ~5% total from each dimension.",
            fn=_centre_crop,
            params={"fraction": 0.05},
            tag="geometry",
        ),
        Transformation(
            name="crop_10pct",
            description="Centre crop removing ~10% total from each dimension.",
            fn=_centre_crop,
            params={"fraction": 0.10},
            tag="geometry",
        ),
        Transformation(
            name="format_convert",
            description="Container swap: PNG→JPEG-95 or JPEG→PNG (direction taken from source).",
            fn=_format_convert,
            tag="format",
        ),
        Transformation(
            name="screenshot_rasterise",
            description="Paste onto a neutral-grey canvas and re-encode as PNG (approximates a screenshot).",
            fn=_screenshot_rasterise,
            params={"padding": 8, "background": (240, 240, 240)},
            tag="pixel",
        ),
        Transformation(
            name="brightness_mild",
            description="Pillow brightness enhance ×1.1, JPEG-95 output.",
            fn=_brightness,
            params={"factor": 1.1},
            tag="pixel",
        ),
    ]
}


#: Minimal set used by ``--smoke``: one control, one metadata operation,
#: one heavy re-encode, one geometric change. Enough to prove the
#: harness end-to-end without touching every entry.
SMOKE_TRANSFORMATIONS: tuple[str, ...] = (
    "original_copy",
    "metadata_stripped",
    "jpeg_q70",
    "resize_50",
)


def transformations_config(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Serialisable spec of every transformation in ``names`` (or all).

    Uses only JSON-safe types so the runner can drop this straight into
    ``experiment_config.json``.
    """
    selected = names or list(TRANSFORMATIONS)
    out: list[dict[str, Any]] = []
    for name in selected:
        t = TRANSFORMATIONS[name]
        out.append({
            "name": t.name,
            "description": t.description,
            "tag": t.tag,
            "params": {
                # Lists rather than tuples so the JSON round-trip is
                # symmetric — reading the config back gives lists.
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in t.params.items()
            },
        })
    return out


__all__ = [
    "Transformation",
    "TRANSFORMATIONS",
    "SMOKE_TRANSFORMATIONS",
    "transformations_config",
]
