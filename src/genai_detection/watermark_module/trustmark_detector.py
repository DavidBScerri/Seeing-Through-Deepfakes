"""
Adobe TrustMark watermark detector — scheme-specific wrapper around the
official ``trustmark`` Python library.

Scope
-----
This detector concerns itself only with TrustMark-family watermarks.
It never speaks about SynthID, Stable Signature, generic
invisible-watermark schemes, or "is this AI-generated?" — those are
either not our business or another module's. See
:data:`~src.genai_detection.watermark_module.models.SCOPE_STATEMENT`.

Result feeds the UI (`/api/analyse`'s ``watermark`` section) as a
standalone Watermark Evidence card. It deliberately does **not** feed
into the fusion formula in this iteration — adding a new fusion weight
requires separate evaluation and David's sign-off per the project's
fusion-formula rule (see ``CLAUDE.md``).

Storage/init contract
---------------------
* The heavy ``trustmark`` import and any model download happen
  **lazily**, on the first analysis call — not at module import time.
  Importing this module never touches the network and never fails
  because of a missing model file.
* Loaded ``TrustMark(model_type=...)`` instances are cached on the
  detector so repeated website requests do not reload the model.
* Model weights live wherever the ``trustmark`` package caches them
  (adjacent to the library install, or wherever the caller has pointed
  a cache directory). This repository never commits TrustMark weights.
* CPU execution is the default; callers can pass a specific device
  through the constructor if they know what they want.
* Every failure path returns a typed :class:`TrustMarkResult` with an
  appropriate :class:`TrustMarkStatus` — the detector never propagates
  its own exceptions to the request handler, so the web app keeps
  serving even when TrustMark is not installed.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from .models import (
    DEFAULT_VARIANT,
    SCHEME_NAME,
    SCOPE_STATEMENT,
    SUPPORTED_VARIANTS,
    TrustMarkResult,
    TrustMarkStatus,
)


# Types the public API accepts. `str` and `Path` are treated as file
# paths, `bytes` as raw image bytes, `PIL.Image.Image` as an already-
# decoded image. Anything else → UNSUPPORTED.
ImageInput = str | Path | bytes | Image.Image


class TrustMarkDetector:
    """
    Adobe TrustMark detector.

    Instantiate once (cheap — no models load until the first
    :meth:`analyse` call) and share the instance across requests. Model
    loads for each variant are memoised in an instance-level cache
    guarded by a lock, so parallel first-hits do not race.

    Parameters
    ----------
    default_variant:
        TrustMark ``model_type`` to use when :meth:`analyse` is not
        given an explicit variant. Must be one of
        :data:`~.models.SUPPORTED_VARIANTS`. Defaults to ``"Q"``
        (matches the ``trustmark`` library default).
    device:
        Torch device string passed straight through to
        ``TrustMark(device=...)``. Defaults to ``"cpu"`` — the upstream
        library interprets an empty string as "auto-select CUDA when
        available", which we do not want for a demo that must run
        reproducibly on a CPU-only box. Pass another value explicitly
        when a specific device is intended.
    verbose:
        Forwarded to ``TrustMark(verbose=...)``. Off by default so the
        web-app log stays clean.
    """

    def __init__(
        self,
        default_variant: str = DEFAULT_VARIANT,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        if default_variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported default TrustMark variant {default_variant!r}. "
                f"Supported: {SUPPORTED_VARIANTS}."
            )
        self.default_variant = default_variant
        self.device = device
        self.verbose = verbose

        # Per-variant model cache. First-hit loads the model; every
        # later request reuses the cached instance. The lock is only
        # taken around loads, not around inference.
        self._model_cache: dict[str, Any] = {}
        self._load_lock = Lock()

        # Set on the first failed load so we don't retry every request
        # once we already know the library or its weights are missing.
        self._unavailable_reason: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        image: ImageInput,
        variant: str | None = None,
    ) -> TrustMarkResult:
        """
        Run TrustMark detection on one image.

        Parameters
        ----------
        image:
            A file path (str or Path), raw image bytes, or an
            already-decoded PIL ``Image``. Anything else returns an
            ``UNSUPPORTED`` result rather than raising.
        variant:
            TrustMark ``model_type`` to try. Defaults to
            ``self.default_variant``.

        Returns
        -------
        TrustMarkResult
            Typed result carrying status, rationale, and — on a positive
            detection — the decoded payload and schema version. The
            detector never raises out of this method; every failure is
            reported through the status field.
        """
        start = time.perf_counter()
        variant = variant or self.default_variant

        # --- Variant sanity check -----------------------------------
        if variant not in SUPPORTED_VARIANTS:
            return self._make_result(
                status=TrustMarkStatus.UNSUPPORTED,
                variant_used=None,
                rationale=(
                    f"Requested TrustMark variant {variant!r} is not one the "
                    f"installed library implements ({', '.join(SUPPORTED_VARIANTS)})."
                ),
                error_details=None,
                elapsed=time.perf_counter() - start,
            )

        # --- Coerce input to a PIL Image ---------------------------
        try:
            pil_image = _coerce_to_pil(image)
        except _UnsupportedInputError as exc:
            return self._make_result(
                status=TrustMarkStatus.UNSUPPORTED,
                variant_used=variant,
                rationale=str(exc),
                error_details=None,
                elapsed=time.perf_counter() - start,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._make_result(
                status=TrustMarkStatus.ERROR,
                variant_used=variant,
                rationale="Could not open the supplied image for TrustMark decoding.",
                error_details=f"{type(exc).__name__}: {exc}",
                elapsed=time.perf_counter() - start,
            )

        # --- Load model (lazy, cached) ------------------------------
        try:
            model = self._get_model(variant)
        except _DetectorUnavailableError as exc:
            return self._make_result(
                status=TrustMarkStatus.DETECTOR_UNAVAILABLE,
                variant_used=variant,
                rationale=(
                    "TrustMark detector is unavailable — the watermark evidence "
                    "card cannot make an assertion about this image. This is NOT "
                    "'no watermark found'."
                ),
                error_details=str(exc),
                elapsed=time.perf_counter() - start,
            )

        # --- Run decode --------------------------------------------
        try:
            secret_pred, detected, schema_version = model.decode(pil_image)
        except Exception as exc:
            return self._make_result(
                status=TrustMarkStatus.ERROR,
                variant_used=variant,
                rationale="TrustMark decoder raised while processing this image.",
                error_details=f"{type(exc).__name__}: {exc}",
                elapsed=time.perf_counter() - start,
            )

        elapsed = time.perf_counter() - start

        if detected:
            payload = secret_pred if isinstance(secret_pred, str) else str(secret_pred)
            return self._make_result(
                status=TrustMarkStatus.DETECTED,
                variant_used=variant,
                rationale=(
                    f"Adobe TrustMark watermark decoded (variant {variant}, "
                    f"schema {schema_version})."
                ),
                error_details=None,
                elapsed=elapsed,
                schema_version=int(schema_version) if schema_version is not None else None,
                payload=payload,
            )

        return self._make_result(
            status=TrustMarkStatus.NOT_DETECTED,
            variant_used=variant,
            rationale=(
                f"No Adobe TrustMark watermark was decoded with variant {variant}. "
                "This does not mean the image is unwatermarked or real — see the "
                "scope statement."
            ),
            error_details=None,
            elapsed=elapsed,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model(self, variant: str) -> Any:
        """
        Return the cached TrustMark instance for ``variant``, loading
        it on first request.

        Raises :class:`_DetectorUnavailableError` when the library is
        missing or the model cannot be loaded. Once a variant has been
        marked unavailable, subsequent calls fail fast without
        re-attempting the import.
        """
        cached = self._model_cache.get(variant)
        if cached is not None:
            return cached

        with self._load_lock:
            cached = self._model_cache.get(variant)
            if cached is not None:
                return cached

            if self._unavailable_reason is not None:
                raise _DetectorUnavailableError(self._unavailable_reason)

            try:
                # Deferred import — importing this module must never
                # require the trustmark package to be installed.
                from trustmark import TrustMark  # type: ignore[import-not-found]
            except Exception as exc:
                reason = f"trustmark library not importable: {type(exc).__name__}: {exc}"
                self._unavailable_reason = reason
                raise _DetectorUnavailableError(reason) from exc

            try:
                # This module only decodes watermarks. Explicitly opt
                # OUT of the watermark-remover and the bbox / localiser
                # stages — they add memory and download surface for
                # capabilities we do not use, and upstream loads them
                # by default (see trustmark/trustmark.py). Documented
                # caveat: upstream may still initialise auxiliary
                # components it does not expose an off-switch for; the
                # constructor arguments above are the extent of what
                # this wrapper can disable through the public API.
                model = TrustMark(
                    model_type=variant,
                    device=self.device,
                    verbose=self.verbose,
                    loadRemover=False,
                    loadBBoxDetector=False,
                )
            except Exception as exc:
                # Model weight download / initialisation failed. Mark
                # the detector unavailable so we do not thrash the
                # network on every request.
                reason = (
                    f"TrustMark({variant!r}) failed to initialise: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._unavailable_reason = reason
                raise _DetectorUnavailableError(reason) from exc

            self._model_cache[variant] = model
            return model

    def _make_result(
        self,
        *,
        status: TrustMarkStatus,
        variant_used: str | None,
        rationale: str,
        error_details: str | None,
        elapsed: float,
        schema_version: int | None = None,
        payload: str | None = None,
    ) -> TrustMarkResult:
        return TrustMarkResult(
            scheme=SCHEME_NAME,
            supported_variants=list(SUPPORTED_VARIANTS),
            variant_used=variant_used,
            status=status,
            detected=(status == TrustMarkStatus.DETECTED),
            schema_version=schema_version,
            payload=payload,
            rationale=rationale,
            error_details=error_details,
            processing_time_seconds=float(elapsed),
            scope_statement=SCOPE_STATEMENT,
        )


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------

class _UnsupportedInputError(Exception):
    """Raised when the caller passes an input shape we can't decode."""


class _DetectorUnavailableError(RuntimeError):
    """Raised inside the detector when the library or model is missing."""


def _coerce_to_pil(image: ImageInput) -> Image.Image:
    """
    Turn a path / bytes / PIL image into a PIL ``Image`` ready for
    TrustMark's decoder.

    The image is converted to RGB (TrustMark works on 3-channel input)
    but **never resized or recompressed** before decoding — the
    watermark lives in the pixels and any pre-scaling would trash the
    signal.
    """
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, (str, Path)):
        pil = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(image))
    else:
        raise _UnsupportedInputError(
            f"Unsupported image input type {type(image).__name__} — "
            "TrustMarkDetector.analyse accepts a file path, PIL Image, or bytes."
        )

    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil


__all__ = ["TrustMarkDetector", "ImageInput"]
