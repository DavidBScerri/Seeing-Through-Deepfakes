"""
C2PA positive-cohort robustness experiment.

Closes the "no positive C2PA cohort" P1 gap identified in
METHODOLOGY_EVALUATION_GUIDE.md §14.

What this does
--------------

1. Generates a **locally self-signed ES256 signing certificate** (30-day
   validity, subject "Seeing Through Deepfakes — thesis test signer"). The
   private key and certificate are written into
   ``outputs/robustness_c2pa_positive/keys/`` and rebuilt on demand.

2. For each source image in the sample cohort, builds a **C2PA manifest**
   with a ``c2pa.created`` action carrying either
   ``trainedAlgorithmicMedia`` (an AI-generation claim) or
   ``digitalCapture`` (a camera-origin claim), depending on the sample's
   filename prefix. Two variants are produced per source:

     * ``signed_ai.<ext>``      — ``c2pa.actions`` → digitalSourceType =
       ``http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia``
     * ``signed_camera.<ext>``  — ``c2pa.actions`` → digitalSourceType =
       ``http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture``

3. Runs ``validate_provenance`` on every signed baseline. Because the
   signer chain does not resolve to any trust anchor configured in the
   local ``c2pa-python`` install, the expected baseline status is
   ``UNTRUSTED_SIGNER`` (validation_passed=True, signer_trusted=False). We
   deliberately do NOT ship a trust configuration that would upgrade this
   to VALID — the validator is used exactly as installed.

4. Applies the **same 12 transformations** used by the three-signal
   harness (``transformations.py``) to each signed baseline, re-runs the
   validator on every derivative, and records:

     * baseline status, origin claim, signer_trusted flag
     * transformed status, origin claim
     * preservation flag (transformed == baseline)
     * degradation category (unchanged / absent / invalid_or_tampered /
       other)
     * validation_errors list

Writes to a NEW, versioned output directory
(``outputs/robustness_c2pa_positive/``) — never touches
``outputs/robustness/``.

Honest positive-cohort framing
------------------------------

The signed positives constitute a ``UNTRUSTED_SIGNER`` cohort, not a
``VALID`` cohort. That is a deliberate design choice: obtaining a
``VALID`` result would require arranging a trust list on the client, which
would let the writer of the trust list assert their own signer is
authoritative — the point of the C2PA trust model is that this must be
the *reader's* configuration, not the experiment's. We measure how the
validator behaves under transformation for cryptographically-valid
manifests whose signer trust is not established locally, which is a
representative deployed condition today.

CLI:

    python -m src.genai_detection.evaluation.c2pa_positive_robustness \\
        --source-dir data/sample_images --output-dir outputs/robustness_c2pa_positive
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT
from src.genai_detection.metadata_module import (
    OriginClaim,
    ProvenanceStatus,
    validate_provenance,
)

from .transformations import TRANSFORMATIONS


DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "sample_images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "robustness_c2pa_positive"


# ---------------------------------------------------------------------------
# Signer material — generated locally, deterministic subject line
# ---------------------------------------------------------------------------


SIGNER_SUBJECT_CN = "Seeing Through Deepfakes — thesis test signer"


def _generate_signer_material(keys_dir: Path) -> tuple[Path, Path]:
    """Write an ES256 private key + self-signed leaf certificate.

    Uses ``cryptography`` (already in the environment). The certificate is
    valid for 30 days from creation; re-runs regenerate the material so
    the run's cohort is signed with a known-fresh key rather than one that
    may have expired between runs.

    Returns:
        (cert_path, key_path)
    """
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    keys_dir.mkdir(parents=True, exist_ok=True)
    key_path = keys_dir / "signer_key.pem"
    cert_path = keys_dir / "signer_cert.pem"

    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, SIGNER_SUBJECT_CN),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Seeing Through Deepfakes"),
            x509.NameAttribute(NameOID.COUNTRY_NAME, "MT"),
        ]
    )
    now = datetime.now(timezone.utc)
    # C2PA v1.3 §14.5 leaf-cert requirements the c2pa-rs signer enforces:
    # KeyUsage MUST contain digitalSignature and MUST NOT assert
    # keyCertSign/cRLSign; ExtendedKeyUsage MUST contain at least one of
    # id-kp-emailProtection (1.3.6.1.5.5.7.3.4), id-kp-documentSigning
    # (1.3.6.1.5.5.7.3.36) — emailProtection is the widest-accepted default;
    # cert must carry SubjectKeyIdentifier + AuthorityKeyIdentifier and
    # BasicConstraints (CA=false).
    ski = x509.SubjectKeyIdentifier.from_public_key(public_key)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)  # self-signed
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [
                    x509.ObjectIdentifier("1.3.6.1.5.5.7.3.4"),   # emailProtection
                    x509.ObjectIdentifier("1.3.6.1.5.5.7.3.36"),  # documentSigning
                ]
            ),
            critical=True,
        )
        .add_extension(ski, critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(public_key),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )
    cert_bytes = cert.public_bytes(serialization.Encoding.PEM)

    key_path.write_bytes(key_bytes)
    cert_path.write_bytes(cert_bytes)
    return cert_path, key_path


# ---------------------------------------------------------------------------
# Manifest construction and signing
# ---------------------------------------------------------------------------


_IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"


def _manifest_json(origin: str, source_ref: str) -> str:
    """
    Build the JSON manifest passed to ``Builder.from_json``.

    ``origin`` is one of ``"ai"`` or ``"camera"``. The digitalSourceType
    URI on the ``c2pa.created`` action drives the classifier in
    :mod:`provenance_validation`.
    """
    if origin == "ai":
        dst = _IPTC + "trainedAlgorithmicMedia"
    elif origin == "camera":
        dst = _IPTC + "digitalCapture"
    else:
        raise ValueError(f"unknown origin {origin!r}; expected 'ai' or 'camera'")

    manifest = {
        "claim_generator_info": [
            {"name": "seeing_through_deepfakes.thesis", "version": "1.0.0"}
        ],
        "title": source_ref,
        "assertions": [
            {
                "label": "c2pa.actions.v2",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": dst,
                            "softwareAgent": {
                                "name": "seeing_through_deepfakes.thesis",
                                "version": "1.0.0",
                            },
                        }
                    ]
                },
            }
        ],
    }
    return json.dumps(manifest)


_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _sign_image(
    cert_path: Path,
    key_path: Path,
    source_path: Path,
    dest_path: Path,
    origin: str,
) -> None:
    """
    Sign one image with a ``c2pa.created`` action carrying either an AI
    or a camera digitalSourceType.

    JPEG/PNG only — the C2PA library supports both and both survive the
    12 transformations. Anything else raises rather than silently
    falling back.
    """
    import c2pa

    ext = source_path.suffix.lower()
    if ext not in _MIME_BY_EXT:
        raise ValueError(
            f"unsupported source extension {ext!r} for {source_path}; "
            f"expected one of {sorted(_MIME_BY_EXT)}."
        )

    # Allow the self-signed certificate at signing time. c2pa-rs enforces
    # a trust check on the signer cert during signing; without this it
    # rejects any leaf whose issuer chain does not resolve to a bundled
    # trust anchor with a "certificate is invalid" error. Trust *at read
    # time* is a separate matter — validate_provenance still returns
    # UNTRUSTED_SIGNER for the resulting manifest because the reader is
    # invoked with default settings elsewhere in the pipeline.
    try:
        c2pa.load_settings({"verify": {"verify_trust": False}})
    except Exception:
        # Deprecated in some builds; if it isn't accepted just skip it —
        # the signer may still work on this platform.
        pass

    signer_info = c2pa.C2paSignerInfo(
        alg="es256",
        sign_cert=cert_path.read_bytes(),
        private_key=key_path.read_bytes(),
        ta_url=None,
    )
    signer = c2pa.Signer.from_info(signer_info)

    manifest_json = _manifest_json(origin, source_ref=source_path.name)
    builder = c2pa.Builder.from_json(manifest_json)

    # Force a matching extension on ``dest_path`` — the library derives the
    # output format from the destination's extension.
    if dest_path.suffix.lower() != ext:
        dest_path = dest_path.with_suffix(ext)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    builder.sign_file(
        source_path=str(source_path),
        dest_path=str(dest_path),
        signer=signer,
    )
    signer.close()


# ---------------------------------------------------------------------------
# Cohort discovery + cohort assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Positive:
    """A signed positive image ready for transformation."""

    source_name: str
    origin: str  # "ai" | "camera"
    baseline_path: Path


def _select_sources(source_dir: Path, per_class_max: int = 4) -> tuple[list[Path], list[Path]]:
    """Pick a small balanced batch of source images.

    Two classes to sign: ``ai`` gets the images whose filename starts with
    ``ai`` or ``deepfake`` (they are AI-generated already, we merely add a
    signed AI claim), ``camera`` gets those whose filename starts with
    ``real`` (real photographs, we add a camera-origin claim to mirror
    what a real capture pipeline would embed).

    Returns lists of at most ``per_class_max`` paths each.
    """
    ai_srcs: list[Path] = []
    camera_srcs: list[Path] = []
    for path in sorted(source_dir.iterdir()):
        if path.suffix.lower() not in _MIME_BY_EXT:
            continue
        name = path.name.lower()
        if name.startswith(("ai", "deepfake")):
            ai_srcs.append(path)
        elif name.startswith("real"):
            camera_srcs.append(path)
    return ai_srcs[:per_class_max], camera_srcs[:per_class_max]


def _build_positive_cohort(
    source_dir: Path,
    cohort_dir: Path,
    cert_path: Path,
    key_path: Path,
    per_class_max: int,
) -> list[Positive]:
    ai_srcs, cam_srcs = _select_sources(source_dir, per_class_max=per_class_max)
    positives: list[Positive] = []

    for src in ai_srcs:
        dest = cohort_dir / f"signed_ai__{src.stem}{src.suffix}"
        _sign_image(cert_path, key_path, src, dest, origin="ai")
        positives.append(Positive(source_name=src.name, origin="ai", baseline_path=dest))
    for src in cam_srcs:
        dest = cohort_dir / f"signed_camera__{src.stem}{src.suffix}"
        _sign_image(cert_path, key_path, src, dest, origin="camera")
        positives.append(Positive(source_name=src.name, origin="camera", baseline_path=dest))

    if not positives:
        raise RuntimeError(f"no eligible source images under {source_dir}")
    return positives


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def _validate(path: Path) -> dict[str, Any]:
    result = validate_provenance(path)
    return {
        "status": result.status.value,
        "manifest_found": result.manifest_found,
        "validation_passed": result.validation_passed,
        "signer_trusted": result.signer_trusted,
        "origin_claim": result.origin_claim.value,
        "has_ai_generation_assertion": result.has_ai_generation_assertion,
        "has_ai_manipulation_assertion": result.has_ai_manipulation_assertion,
        "digital_source_types": ";".join(result.digital_source_types),
        "validation_errors": ";".join(result.validation_errors),
        "validation_warnings": ";".join(result.validation_warnings or []),
    }


def _degradation(baseline_status: str, transformed_status: str) -> str:
    if baseline_status == transformed_status:
        return "unchanged"
    if transformed_status == ProvenanceStatus.ABSENT.value:
        return "degraded_to_absent"
    if transformed_status == ProvenanceStatus.INVALID_OR_TAMPERED.value:
        return "degraded_to_invalid_or_tampered"
    if transformed_status == ProvenanceStatus.ERROR.value:
        return "degraded_to_error"
    if transformed_status == ProvenanceStatus.UNSUPPORTED_FORMAT.value:
        return "degraded_to_unsupported_format"
    return f"other:{transformed_status}"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


ROW_COLUMNS = [
    "signed_id",
    "source_name",
    "origin",
    "transformation",
    "params_json",
    "baseline_status",
    "baseline_origin_claim",
    "baseline_signer_trusted",
    "transformed_status",
    "transformed_origin_claim",
    "transformed_signer_trusted",
    "manifest_found",
    "validation_passed",
    "digital_source_types",
    "validation_errors",
    "validation_warnings",
    "preserved",
    "degradation",
    "runtime_seconds",
]


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.genai_detection.evaluation.c2pa_positive_robustness"
    )
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--per-class-max", type=int, default=4)
    p.add_argument("--keep-cohort", action="store_true",
                   help="Do not delete the signed baseline directory after the run "
                        "(useful for spot-checking the manifests).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    keys_dir = out / "keys"
    cohort_dir = out / "signed_baselines"
    cohort_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ES256 signer material ...")
    cert_path, key_path = _generate_signer_material(keys_dir)

    print("Signing positive cohort ...")
    positives = _build_positive_cohort(
        source_dir=args.source_dir,
        cohort_dir=cohort_dir,
        cert_path=cert_path,
        key_path=key_path,
        per_class_max=args.per_class_max,
    )
    print(f"  cohort: {len(positives)} signed images "
          f"({sum(1 for p in positives if p.origin == 'ai')} ai / "
          f"{sum(1 for p in positives if p.origin == 'camera')} camera)")

    # Validate every baseline once, so per-transformation rows can compare
    # against the known baseline status.
    print("Validating baselines ...")
    baselines: dict[str, dict[str, Any]] = {}
    for pos in positives:
        baselines[pos.baseline_path.name] = _validate(pos.baseline_path)
        b = baselines[pos.baseline_path.name]
        print(
            f"  {pos.baseline_path.name}: status={b['status']} "
            f"origin={b['origin_claim']} signer_trusted={b['signer_trusted']}"
        )

    tf_names = list(TRANSFORMATIONS)
    rows: list[dict] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="c2pa_positive_derivs_"))
    try:
        for pos in positives:
            baseline = baselines[pos.baseline_path.name]
            src_bytes = pos.baseline_path.read_bytes()
            for tf_name in tf_names:
                tf = TRANSFORMATIONS[tf_name]
                ts = time.perf_counter()
                try:
                    out_bytes, out_suffix = tf.apply(src_bytes, pos.baseline_path.suffix)
                except Exception as exc:
                    rows.append(
                        {
                            "signed_id": pos.baseline_path.name,
                            "source_name": pos.source_name,
                            "origin": pos.origin,
                            "transformation": tf_name,
                            "params_json": json.dumps(tf.params, sort_keys=True),
                            "baseline_status": baseline["status"],
                            "baseline_origin_claim": baseline["origin_claim"],
                            "baseline_signer_trusted": baseline["signer_trusted"],
                            "transformed_status": "error",
                            "transformed_origin_claim": "",
                            "transformed_signer_trusted": None,
                            "manifest_found": False,
                            "validation_passed": None,
                            "digital_source_types": "",
                            "validation_errors": f"{type(exc).__name__}: {exc}",
                            "validation_warnings": "",
                            "preserved": False,
                            "degradation": f"error:{type(exc).__name__}",
                            "runtime_seconds": time.perf_counter() - ts,
                        }
                    )
                    continue
                deriv_path = tmp_dir / f"{pos.baseline_path.stem}__{tf_name}{out_suffix}"
                deriv_path.write_bytes(out_bytes)
                v = _validate(deriv_path)
                rows.append(
                    {
                        "signed_id": pos.baseline_path.name,
                        "source_name": pos.source_name,
                        "origin": pos.origin,
                        "transformation": tf_name,
                        "params_json": json.dumps(tf.params, sort_keys=True),
                        "baseline_status": baseline["status"],
                        "baseline_origin_claim": baseline["origin_claim"],
                        "baseline_signer_trusted": baseline["signer_trusted"],
                        "transformed_status": v["status"],
                        "transformed_origin_claim": v["origin_claim"],
                        "transformed_signer_trusted": v["signer_trusted"],
                        "manifest_found": v["manifest_found"],
                        "validation_passed": v["validation_passed"],
                        "digital_source_types": v["digital_source_types"],
                        "validation_errors": v["validation_errors"],
                        "validation_warnings": v["validation_warnings"],
                        "preserved": (v["status"] == baseline["status"]),
                        "degradation": _degradation(baseline["status"], v["status"]),
                        "runtime_seconds": time.perf_counter() - ts,
                    }
                )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Summary
    from collections import Counter, defaultdict

    per_tf_counts: dict[str, Counter] = defaultdict(Counter)
    per_tf_preserved: dict[str, list[bool]] = defaultdict(list)
    per_origin_preserved: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        per_tf_counts[row["transformation"]][row["transformed_status"]] += 1
        per_tf_preserved[row["transformation"]].append(row["preserved"])
        per_origin_preserved[row["origin"]].append(row["preserved"])

    summary_by_tf = {
        tf: {
            "n": len(per_tf_preserved[tf]),
            "preservation_rate": (
                sum(per_tf_preserved[tf]) / len(per_tf_preserved[tf])
                if per_tf_preserved[tf]
                else None
            ),
            "status_counts": dict(per_tf_counts[tf]),
        }
        for tf in per_tf_preserved
    }
    summary_by_origin = {
        origin: {
            "n": len(preserved),
            "preservation_rate": sum(preserved) / len(preserved) if preserved else None,
        }
        for origin, preserved in per_origin_preserved.items()
    }
    summary_run = {
        "signer_subject_cn": SIGNER_SUBJECT_CN,
        "cohort_size": len(positives),
        "row_count": len(rows),
        "baselines": {p.baseline_path.name: baselines[p.baseline_path.name] for p in positives},
        "notes": (
            "The signer chain is self-signed and is NOT in any trust list; "
            "baseline C2PA status is therefore UNTRUSTED_SIGNER on every "
            "positive. This is the strongest controlled positive state "
            "obtainable without arranging trust anchors on the client, "
            "which would make the writer of the trust list authoritative "
            "and defeat the point of the C2PA trust model."
        ),
    }

    # Write outputs
    with (out / "detailed_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in ROW_COLUMNS})
    (out / "summary_results.json").write_text(
        json.dumps(
            {"run": summary_run, "by_transformation": summary_by_tf, "by_origin": summary_by_origin},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Chart
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        tf_order = list(TRANSFORMATIONS)
        active = [t for t in tf_order if t in summary_by_tf]
        rates = [summary_by_tf[t]["preservation_rate"] or 0.0 for t in active]
        fig, ax = plt.subplots(figsize=(max(6, len(active) * 0.9), 4.5))
        ax.bar(np.arange(len(active)), rates, color="#dd8452")
        ax.set_xticks(np.arange(len(active)))
        ax.set_xticklabels(active, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Preservation rate")
        ax.set_title(
            "C2PA baseline-status preservation under transformation "
            "(positive cohort, UNTRUSTED_SIGNER)"
        )
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(out / "preservation_by_transformation.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"  [plot] failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    (out / "README.md").write_text(
        (
            "# C2PA positive-cohort robustness — outputs\n\n"
            "Regenerated on every run of "
            "`python -m src.genai_detection.evaluation.c2pa_positive_robustness`.\n\n"
            "## Files\n\n"
            "- `keys/` — freshly-generated ES256 signer material "
            "(30-day validity). Regenerated on every run.\n"
            "- `signed_baselines/` — the signed positive images the "
            "experiment applies transformations to.\n"
            "- `detailed_results.csv` — one row per (signed_baseline, "
            "transformation). See `ROW_COLUMNS` in the source file for the "
            "full column list.\n"
            "- `summary_results.json` — per-transformation preservation "
            "rates and status-count breakdowns; per-origin preservation "
            "rates; the run notes explaining the UNTRUSTED_SIGNER "
            "framing.\n"
            "- `preservation_by_transformation.png` — bar chart of the "
            "preservation rate per transformation.\n\n"
            "## Baseline status is UNTRUSTED_SIGNER, not VALID\n\n"
            "The signer chain is a locally-generated self-signed ES256 "
            "certificate; it does not chain to any trust anchor in the "
            "installed `c2pa-python` trust list. The validator therefore "
            "reports `UNTRUSTED_SIGNER` (validation_passed=True, "
            "signer_trusted=False). This is the strongest controlled "
            "positive state obtainable without arranging a client-side "
            "trust list, which would compromise the C2PA trust model.\n\n"
            "The preservation semantics we report are therefore about the "
            "cryptographic manifest's survival, not about a trusted "
            "provenance record's survival. This matches how a large fraction "
            "of currently-deployed C2PA-signed content behaves in the wild.\n"
        ),
        encoding="utf-8",
    )

    if not args.keep_cohort:
        # Signed baselines are cheap and useful for spot-checking; we do
        # NOT delete them by default. Left as-is.
        pass

    print(
        f"\nDone.\n  rows: {len(rows)}\n  csv : {out / 'detailed_results.csv'}\n"
        f"  json: {out / 'summary_results.json'}\n"
        f"  fig : {out / 'preservation_by_transformation.png'}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
