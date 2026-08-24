"""
Small command-line front end for the hash module.

Three subcommands, all deliberately narrow:

* ``hash IMAGE`` — print the streaming SHA-256 digest of ``IMAGE``.
* ``register IMAGE --label <label> [...]`` — record the digest of
  ``IMAGE`` in the runtime registry. **Never copies the image.**
* ``lookup IMAGE`` — hash ``IMAGE`` and look the digest up.

The CLI reads the same
:data:`~.registry.RUNTIME_REGISTRY_ENV` environment variable and
respects the same ``--registry`` override the library API uses, so
scripted usage behaves the same as the web app.

Run as::

    python -m src.genai_detection.hash_module.cli hash path/to/img.png
    python -m src.genai_detection.hash_module.cli register path/to/img.png \\
        --label ai_generated --provider example --model example-v1
    python -m src.genai_detection.hash_module.cli lookup path/to/img.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import HashLookupStatus, HashRecord, OriginLabel, SCOPE_STATEMENT
from .registry import (
    DEFAULT_RUNTIME_REGISTRY_PATH,
    HashRegistry,
    HashRegistryConflictError,
    RUNTIME_REGISTRY_ENV,
    load_registry,
)
from .sha256_hasher import sha256_file


# Exit codes small enough that shell wrappers can rely on them.
EXIT_OK = 0
EXIT_NO_MATCH = 1          # lookup succeeded but the digest is not registered
EXIT_ERROR = 2             # user error (bad arg, missing file, …)
EXIT_REGISTRY_UNAVAILABLE = 3
EXIT_REGISTRY_INVALID = 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.genai_detection.hash_module.cli",
        description=(
            "SHA-256 byte-exact hashing + tiny hash-registry lookup. A "
            "SHA-256 digest is byte-exact, not perceptual — recompression, "
            "metadata edits, or any byte change normally alter the digest. "
            "The registry stores only digests and small descriptive "
            "metadata; no image content is ever recorded."
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help=(
            "Optional explicit registry path. Overrides the "
            f"{RUNTIME_REGISTRY_ENV} environment variable. When neither "
            f"is set, the CLI falls back to {DEFAULT_RUNTIME_REGISTRY_PATH}."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON on stdout instead of the default human summary.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_p = subparsers.add_parser(
        "hash", help="Compute the SHA-256 digest of one image file."
    )
    hash_p.add_argument("image", type=Path, help="Path to the file to hash.")

    reg_p = subparsers.add_parser(
        "register",
        help=(
            "Record the SHA-256 digest of one image in the runtime registry. "
            "The image itself is NOT copied or modified."
        ),
    )
    reg_p.add_argument("image", type=Path, help="Path to the file to register.")
    reg_p.add_argument(
        "--label",
        required=True,
        choices=[o.value for o in OriginLabel],
        help="Closed-vocabulary origin claim recorded with the digest.",
    )
    reg_p.add_argument("--provider", default=None, help="Optional provider string (e.g. 'openai').")
    reg_p.add_argument("--model", default=None, help="Optional model / version string.")
    reg_p.add_argument("--source-reference", default=None, help="Optional non-sensitive reference (dataset id, ticket, URL).")
    reg_p.add_argument("--notes", default=None, help="Optional short free-text notes. No binary data.")
    reg_p.add_argument(
        "--replace",
        action="store_true",
        help="Allow overwriting an existing record for the same digest.",
    )

    lookup_p = subparsers.add_parser(
        "lookup",
        help="Compute the SHA-256 digest of one image and look it up in the registry.",
    )
    lookup_p.add_argument("image", type=Path, help="Path to the file to look up.")

    return parser


def _emit(payload: dict, *, as_json: bool) -> None:
    if as_json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        # Human summary: key: value, one per line, in a stable order.
        for k in sorted(payload):
            v = payload[k]
            sys.stdout.write(f"{k}: {v}\n")


def _cmd_hash(args: argparse.Namespace) -> int:
    if not args.image.exists():
        print(f"error: file not found: {args.image}", file=sys.stderr)
        return EXIT_ERROR
    digest = sha256_file(args.image)
    _emit(
        {"sha256": digest, "path": str(args.image)},
        as_json=args.json,
    )
    return EXIT_OK


def _cmd_register(args: argparse.Namespace) -> int:
    if not args.image.exists():
        print(f"error: file not found: {args.image}", file=sys.stderr)
        return EXIT_ERROR

    digest = sha256_file(args.image)
    record = HashRecord(
        sha256=digest,
        origin_label=OriginLabel(args.label),
        provider=args.provider,
        model=args.model,
        source_reference=args.source_reference,
        notes=args.notes,
    )

    # Registration must succeed even when the registry file does not
    # exist yet — we create a fresh empty one at the resolved path.
    try:
        registry = HashRegistry(args.registry)
    except FileNotFoundError:
        # Bootstrap an empty registry at the resolved path by writing
        # the single new record through the atomic write path.
        from .registry import _atomic_write_registry, _resolve_path

        resolved = _resolve_path(args.registry)
        if resolved is None:
            print("error: no registry path could be resolved", file=sys.stderr)
            return EXIT_ERROR
        _atomic_write_registry(resolved, {digest: record})
        _emit(
            {
                "sha256": digest,
                "registry": str(resolved),
                "created": True,
                "label": record.origin_label.value,
            },
            as_json=args.json,
        )
        return EXIT_OK

    try:
        registry.register(record, allow_replace=args.replace)
    except HashRegistryConflictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    _emit(
        {
            "sha256": digest,
            "registry": str(registry.path),
            "created": False,
            "label": record.origin_label.value,
        },
        as_json=args.json,
    )
    return EXIT_OK


def _cmd_lookup(args: argparse.Namespace) -> int:
    if not args.image.exists():
        print(f"error: file not found: {args.image}", file=sys.stderr)
        return EXIT_ERROR

    digest = sha256_file(args.image)
    loaded = load_registry(args.registry)

    if isinstance(loaded, HashRegistry):
        result = loaded.lookup(digest)
    else:
        # Failure result from load_registry — patch in the digest we
        # just computed so the caller sees which file was checked.
        result = loaded.model_copy(update={"sha256": digest})

    payload = {
        "sha256": result.sha256,
        "status": result.status.value,
        "registry_available": result.registry_available,
        "registry_path": result.registry_path,
        "registry_entry_count": result.registry_entry_count,
        "rationale": result.rationale,
        "match": result.match.model_dump(exclude_none=True) if result.match else None,
        "scope_statement": SCOPE_STATEMENT,
        "error_details": result.error_details,
    }
    _emit(payload, as_json=args.json)

    if result.status == HashLookupStatus.EXACT_MATCH:
        return EXIT_OK
    if result.status == HashLookupStatus.NO_MATCH:
        return EXIT_NO_MATCH
    if result.status == HashLookupStatus.REGISTRY_UNAVAILABLE:
        return EXIT_REGISTRY_UNAVAILABLE
    if result.status == HashLookupStatus.INVALID_REGISTRY:
        return EXIT_REGISTRY_INVALID
    return EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "hash":
        return _cmd_hash(args)
    if args.command == "register":
        return _cmd_register(args)
    if args.command == "lookup":
        return _cmd_lookup(args)
    parser.error(f"unknown command {args.command!r}")
    return EXIT_ERROR  # unreachable, argparse exits on error


if __name__ == "__main__":  # pragma: no cover - executable entry
    raise SystemExit(main())
