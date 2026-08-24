"""
Tests for the SHA-256 hash module (src/genai_detection/hash_module/).

Covers the pieces the rest of the pipeline (and the thesis defence) rely
on:

  * Canonical SHA-256 digests for the standard vectors, both from bytes
    and from a streamed file read (byte-hashing and file-hashing must
    agree bit-for-bit).
  * Registry lookup: exact match, no match, malformed digest, unavailable
    registry, malformed JSON, invalid records, duplicates, size cap.
  * Registration is atomic and never copies the underlying image.
  * A validation failure never overwrites a valid on-disk registry.
  * Environment-variable override for the runtime path.
  * The small CLI in :mod:`~.cli` exposes hash / register / lookup
    without copying the image.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.genai_detection.hash_module import (
    DIGEST_LENGTH,
    HashLookupStatus,
    HashRecord,
    HashRegistry,
    HashRegistryConflictError,
    HashRegistryInvalidError,
    OriginLabel,
    RUNTIME_REGISTRY_ENV,
    SCHEME_NAME,
    SCOPE_STATEMENT,
    is_valid_sha256_hex,
    load_registry,
    normalise_digest,
    sha256_bytes,
    sha256_file,
)
from src.genai_detection.hash_module import cli as hash_cli
from src.genai_detection.hash_module import registry as registry_mod


# ---------------------------------------------------------------------------
# Standard SHA-256 test vectors (FIPS-180-2 Appendix B, plus empty input).
# ---------------------------------------------------------------------------

STANDARD_VECTORS: list[tuple[bytes, str]] = [
    (
        b"",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    (
        b"abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    ),
    (
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    ),
]


# ---------------------------------------------------------------------------
# Digest primitives
# ---------------------------------------------------------------------------


class TestSha256Bytes:
    @pytest.mark.parametrize("data, expected", STANDARD_VECTORS)
    def test_matches_standard_vectors(self, data, expected):
        assert sha256_bytes(data) == expected

    def test_result_is_canonical_form(self):
        digest = sha256_bytes(b"hello")
        assert len(digest) == DIGEST_LENGTH
        assert digest == digest.lower()
        assert is_valid_sha256_hex(digest)

    def test_accepts_bytearray_and_memoryview(self):
        expected = sha256_bytes(b"hello world")
        assert sha256_bytes(bytearray(b"hello world")) == expected
        assert sha256_bytes(memoryview(b"hello world")) == expected

    def test_rejects_str(self):
        with pytest.raises(TypeError):
            sha256_bytes("hello")  # type: ignore[arg-type]


class TestSha256File:
    @pytest.mark.parametrize("data, expected", STANDARD_VECTORS)
    def test_streamed_file_agrees_with_bytes(self, tmp_path, data, expected):
        p = tmp_path / "vec.bin"
        p.write_bytes(data)
        assert sha256_file(p) == expected
        assert sha256_file(p) == sha256_bytes(data)

    def test_streamed_and_bytes_agree_across_chunk_sizes_on_multi_mb_input(self, tmp_path):
        # 3 MiB of pseudo-random bytes so we cross more than one chunk
        # boundary in every configuration below.
        data = (b"seed-abcdef01234567" * 200_000)[: 3 * 1024 * 1024]
        p = tmp_path / "big.bin"
        p.write_bytes(data)
        expected = sha256_bytes(data)
        for chunk in (1, 7, 4096, 1024 * 1024):
            assert sha256_file(p, chunk_size=chunk) == expected

    def test_rejects_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sha256_file(tmp_path / "does-not-exist.bin")

    def test_rejects_directory(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            sha256_file(tmp_path)

    def test_rejects_bad_chunk_size(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"data")
        for bad in (0, -1, 1.5, "big"):
            with pytest.raises(ValueError):
                sha256_file(p, chunk_size=bad)  # type: ignore[arg-type]


class TestNormaliseDigest:
    def test_lowercases(self):
        d = "E" * DIGEST_LENGTH
        assert normalise_digest(d) == "e" * DIGEST_LENGTH

    def test_strips_whitespace(self):
        d = "a" * DIGEST_LENGTH
        assert normalise_digest(f"  {d}\n") == d

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "abc",
            "g" * DIGEST_LENGTH,
            "a" * (DIGEST_LENGTH - 1),
            "a" * (DIGEST_LENGTH + 1),
        ],
    )
    def test_rejects_non_canonical(self, bad):
        with pytest.raises(ValueError):
            normalise_digest(bad)


class TestIsValidSha256Hex:
    def test_accepts_canonical(self):
        assert is_valid_sha256_hex("0" * DIGEST_LENGTH)
        assert is_valid_sha256_hex("abcdef" * 10 + "abcd")

    def test_rejects_upper(self):
        assert not is_valid_sha256_hex("A" * DIGEST_LENGTH)

    def test_rejects_short(self):
        assert not is_valid_sha256_hex("a" * (DIGEST_LENGTH - 1))

    def test_rejects_non_string(self):
        assert not is_valid_sha256_hex(b"a" * DIGEST_LENGTH)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HashRecord + closed schema
# ---------------------------------------------------------------------------


class TestHashRecordSchema:
    def _minimal(self, **kw):
        base = {
            "sha256": "a" * DIGEST_LENGTH,
            "origin_label": OriginLabel.UNKNOWN,
        }
        base.update(kw)
        return HashRecord(**base)

    def test_minimum_valid_record(self):
        r = self._minimal()
        assert r.provider is None and r.notes is None

    def test_rejects_extra_fields(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            HashRecord(
                sha256="a" * DIGEST_LENGTH,
                origin_label=OriginLabel.UNKNOWN,
                thumbnail_b64="not-allowed",  # type: ignore[call-arg]
            )

    def test_rejects_malformed_digest(self):
        with pytest.raises(Exception):
            HashRecord(sha256="not-a-digest", origin_label=OriginLabel.UNKNOWN)

    def test_caps_long_string_fields(self):
        with pytest.raises(Exception):
            self._minimal(notes="x" * 5000)


# ---------------------------------------------------------------------------
# HashRegistry — load / lookup / register / remove
# ---------------------------------------------------------------------------


def _write_registry(path: Path, records: list[dict], envelope: bool = True) -> None:
    if envelope:
        path.write_text(json.dumps({"records": records}, indent=2))
    else:
        path.write_text(json.dumps(records, indent=2))


class TestRegistryLoad:
    def test_missing_file_becomes_unavailable_result(self, tmp_path):
        result = load_registry(tmp_path / "missing.json")
        assert not isinstance(result, HashRegistry)
        assert result.status == HashLookupStatus.REGISTRY_UNAVAILABLE
        assert result.registry_available is False
        assert result.scope_statement == SCOPE_STATEMENT

    def test_valid_registry_loads(self, tmp_path):
        p = tmp_path / "r.json"
        _write_registry(
            p,
            [
                {
                    "sha256": "b" * DIGEST_LENGTH,
                    "origin_label": "ai_generated",
                    "provider": "example",
                }
            ],
        )
        reg = load_registry(p)
        assert isinstance(reg, HashRegistry)
        assert len(reg) == 1

    def test_bare_list_root_is_accepted(self, tmp_path):
        p = tmp_path / "r.json"
        _write_registry(
            p,
            [{"sha256": "b" * DIGEST_LENGTH, "origin_label": "unknown"}],
            envelope=False,
        )
        reg = load_registry(p)
        assert isinstance(reg, HashRegistry)
        assert len(reg) == 1

    def test_malformed_json_is_invalid_registry(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text("{not: valid json,")
        result = load_registry(p)
        assert not isinstance(result, HashRegistry)
        assert result.status == HashLookupStatus.INVALID_REGISTRY
        assert "not valid JSON" in (result.error_details or "")

    def test_wrong_root_type_is_invalid_registry(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text('"just a string"')
        result = load_registry(p)
        assert not isinstance(result, HashRegistry)
        assert result.status == HashLookupStatus.INVALID_REGISTRY

    def test_invalid_record_digest_rejected(self, tmp_path):
        p = tmp_path / "r.json"
        _write_registry(p, [{"sha256": "notadigest", "origin_label": "unknown"}])
        result = load_registry(p)
        assert not isinstance(result, HashRegistry)
        assert result.status == HashLookupStatus.INVALID_REGISTRY

    def test_extra_field_in_record_rejected(self, tmp_path):
        p = tmp_path / "r.json"
        _write_registry(
            p,
            [
                {
                    "sha256": "b" * DIGEST_LENGTH,
                    "origin_label": "unknown",
                    "image_bytes_b64": "AAAA",  # closed schema — must fail
                }
            ],
        )
        result = load_registry(p)
        assert not isinstance(result, HashRegistry)
        assert result.status == HashLookupStatus.INVALID_REGISTRY

    def test_duplicate_digest_rejected(self, tmp_path):
        p = tmp_path / "r.json"
        _write_registry(
            p,
            [
                {"sha256": "b" * DIGEST_LENGTH, "origin_label": "unknown"},
                {"sha256": "b" * DIGEST_LENGTH, "origin_label": "ai_generated"},
            ],
        )
        result = load_registry(p)
        assert not isinstance(result, HashRegistry)
        assert result.status == HashLookupStatus.INVALID_REGISTRY
        assert "duplicate" in (result.error_details or "").lower()

    def test_env_var_override(self, tmp_path, monkeypatch):
        p = tmp_path / "env_reg.json"
        _write_registry(p, [])
        monkeypatch.setenv(RUNTIME_REGISTRY_ENV, str(p))
        reg = load_registry()  # no explicit arg
        assert isinstance(reg, HashRegistry)
        assert reg.path == p

    def test_oversize_registry_rejected(self, tmp_path):
        p = tmp_path / "big.json"
        # Fill just over the cap with padding inside a "notes" field
        # embedded in the envelope so JSON parses fine at first blush;
        # the size check runs before the parser.
        padding = "x" * (11 * 1024 * 1024)
        p.write_text('{"pad": "' + padding + '"}')
        result = load_registry(p)
        assert not isinstance(result, HashRegistry)
        assert result.status == HashLookupStatus.INVALID_REGISTRY
        assert "refusing to load" in (result.error_details or "")


class TestRegistryLookup:
    def test_exact_match(self, tmp_path):
        d = sha256_bytes(b"hello registered image")
        p = tmp_path / "r.json"
        _write_registry(
            p,
            [
                {
                    "sha256": d,
                    "origin_label": "ai_generated",
                    "provider": "example",
                    "model": "v1",
                }
            ],
        )
        reg = load_registry(p)
        assert isinstance(reg, HashRegistry)
        result = reg.lookup(d)
        assert result.status == HashLookupStatus.EXACT_MATCH
        assert result.match is not None
        assert result.match.provider == "example"
        assert result.registry_available is True
        assert "ai_generated" in result.rationale

    def test_no_match_is_inconclusive_language(self, tmp_path):
        p = tmp_path / "r.json"
        _write_registry(
            p,
            [{"sha256": "b" * DIGEST_LENGTH, "origin_label": "unknown"}],
        )
        reg = load_registry(p)
        assert isinstance(reg, HashRegistry)
        result = reg.lookup("a" * DIGEST_LENGTH)
        assert result.status == HashLookupStatus.NO_MATCH
        assert result.match is None
        # Must not claim "real" / "not AI".
        combined = (result.rationale + " " + result.scope_statement).lower()
        assert "inconclusive" in combined
        assert "not real" not in combined
        # Common false-negative phrasings must be absent.
        assert "not ai" not in result.rationale.lower()

    def test_malformed_digest_lookup_becomes_error(self, tmp_path):
        p = tmp_path / "r.json"
        _write_registry(p, [])
        reg = load_registry(p)
        assert isinstance(reg, HashRegistry)
        result = reg.lookup("not-a-digest")
        assert result.status == HashLookupStatus.ERROR
        assert result.error_details is not None

    def test_lookup_normalises_case(self, tmp_path):
        d_lower = "c" * DIGEST_LENGTH
        p = tmp_path / "r.json"
        _write_registry(p, [{"sha256": d_lower, "origin_label": "unknown"}])
        reg = load_registry(p)
        assert isinstance(reg, HashRegistry)
        result = reg.lookup("C" * DIGEST_LENGTH)
        assert result.status == HashLookupStatus.EXACT_MATCH


# ---------------------------------------------------------------------------
# HashRegistry — register / remove semantics
# ---------------------------------------------------------------------------


class TestRegistryRegister:
    def _empty(self, tmp_path) -> tuple[HashRegistry, Path]:
        p = tmp_path / "r.json"
        _write_registry(p, [])
        reg = load_registry(p)
        assert isinstance(reg, HashRegistry)
        return reg, p

    def test_idempotent_reregister(self, tmp_path):
        reg, _ = self._empty(tmp_path)
        r = HashRecord(sha256="a" * DIGEST_LENGTH, origin_label=OriginLabel.UNKNOWN)
        reg.register(r)
        reg.register(r)  # identical — no-op
        assert len(reg) == 1

    def test_conflicting_reregister_raises(self, tmp_path):
        reg, _ = self._empty(tmp_path)
        d = "a" * DIGEST_LENGTH
        reg.register(HashRecord(sha256=d, origin_label=OriginLabel.UNKNOWN))
        with pytest.raises(HashRegistryConflictError):
            reg.register(
                HashRecord(sha256=d, origin_label=OriginLabel.AI_GENERATED)
            )
        # Still the original label.
        assert reg.get(d).origin_label == OriginLabel.UNKNOWN

    def test_allow_replace_overrides_conflict(self, tmp_path):
        reg, _ = self._empty(tmp_path)
        d = "a" * DIGEST_LENGTH
        reg.register(HashRecord(sha256=d, origin_label=OriginLabel.UNKNOWN))
        reg.register(
            HashRecord(sha256=d, origin_label=OriginLabel.AI_GENERATED),
            allow_replace=True,
        )
        assert reg.get(d).origin_label == OriginLabel.AI_GENERATED

    def test_register_writes_atomically(self, tmp_path, monkeypatch):
        """os.replace failing mid-write must not leave a corrupt registry
        or a stray temp file on disk."""
        reg, path = self._empty(tmp_path)
        r = HashRecord(sha256="a" * DIGEST_LENGTH, origin_label=OriginLabel.UNKNOWN)

        boom_called = {"n": 0}
        real_replace = os.replace

        def boom(src, dst):
            boom_called["n"] += 1
            raise OSError("simulated replace failure")

        monkeypatch.setattr(registry_mod.os, "replace", boom)
        with pytest.raises(OSError):
            reg.register(r)

        # Restore for cleanup assertions.
        monkeypatch.setattr(registry_mod.os, "replace", real_replace)

        # The on-disk file is still the original (empty) registry.
        on_disk = json.loads(path.read_text())
        assert on_disk["records"] == []
        # And no orphan .tmp files.
        stray = list(path.parent.glob(".registry.*.tmp"))
        assert stray == [], f"orphan temp files after failed atomic write: {stray}"

    def test_remove_existing_and_missing(self, tmp_path):
        reg, _ = self._empty(tmp_path)
        d = "d" * DIGEST_LENGTH
        reg.register(HashRecord(sha256=d, origin_label=OriginLabel.UNKNOWN))
        assert reg.remove(d) is True
        assert reg.remove(d) is False


class TestValidationFailureDoesNotOverwrite:
    def test_load_of_invalid_file_does_not_touch_disk(self, tmp_path):
        p = tmp_path / "r.json"
        original = "{not: valid json"
        p.write_text(original)
        result = load_registry(p)
        assert not isinstance(result, HashRegistry)
        # Bytes on disk unchanged.
        assert p.read_text() == original


# ---------------------------------------------------------------------------
# Registration must not copy the image file
# ---------------------------------------------------------------------------


class TestRegistrationDoesNotCopyImage:
    def test_no_new_files_appear_in_image_directory(self, tmp_path):
        # Image lives in one dir, registry lives in another.
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        image_path = image_dir / "photo.png"
        image_bytes = _tiny_png_bytes()
        image_path.write_bytes(image_bytes)

        registry_dir = tmp_path / "reg"
        registry_dir.mkdir()
        registry_path = registry_dir / "r.json"
        _write_registry(registry_path, [])

        # Snapshot both directories.
        before_images = {p.name: p.stat() for p in image_dir.iterdir()}
        before_registry = {p.name for p in registry_dir.iterdir()}

        reg = load_registry(registry_path)
        assert isinstance(reg, HashRegistry)
        reg.register(
            HashRecord(
                sha256=sha256_bytes(image_bytes),
                origin_label=OriginLabel.AI_GENERATED,
                provider="test",
            )
        )

        after_images = {p.name: p.stat() for p in image_dir.iterdir()}
        # Same image files, same mtimes / sizes.
        assert set(before_images) == set(after_images)
        for name, stat in before_images.items():
            assert after_images[name].st_size == stat.st_size
            assert after_images[name].st_mtime == stat.st_mtime

        # Registry dir may have grown by exactly the registry file (no
        # image copies, no thumbnail dumps).
        after_registry = {p.name for p in registry_dir.iterdir()}
        assert after_registry == before_registry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _tiny_png_bytes(colour=(200, 40, 40)) -> bytes:
    from PIL import Image  # local import to avoid hard test-collection dep
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(buf, format="PNG")
    return buf.getvalue()


class TestCLI:
    def test_hash_prints_digest(self, tmp_path, capsys):
        p = tmp_path / "img.png"
        p.write_bytes(_tiny_png_bytes())
        rc = hash_cli.main(["--json", "hash", str(p)])
        out = capsys.readouterr().out
        assert rc == hash_cli.EXIT_OK
        payload = json.loads(out)
        assert payload["sha256"] == sha256_bytes(_tiny_png_bytes())

    def test_register_then_lookup_flow(self, tmp_path, capsys):
        img = tmp_path / "img.png"
        img.write_bytes(_tiny_png_bytes())
        reg = tmp_path / "r.json"

        rc = hash_cli.main(
            [
                "--registry", str(reg), "--json",
                "register", str(img),
                "--label", "ai_generated",
                "--provider", "test",
            ]
        )
        assert rc == hash_cli.EXIT_OK
        capsys.readouterr()  # drop register output

        rc = hash_cli.main(
            ["--registry", str(reg), "--json", "lookup", str(img)]
        )
        out = capsys.readouterr().out
        assert rc == hash_cli.EXIT_OK  # exact match
        payload = json.loads(out)
        assert payload["status"] == "exact_match"
        assert payload["match"]["origin_label"] == "ai_generated"

    def test_lookup_no_match_exits_1(self, tmp_path, capsys):
        img = tmp_path / "img.png"
        img.write_bytes(_tiny_png_bytes())
        reg = tmp_path / "r.json"
        _write_registry(reg, [])
        rc = hash_cli.main(["--registry", str(reg), "--json", "lookup", str(img)])
        capsys.readouterr()
        assert rc == hash_cli.EXIT_NO_MATCH

    def test_lookup_missing_registry_exits_registry_unavailable(self, tmp_path, capsys):
        img = tmp_path / "img.png"
        img.write_bytes(_tiny_png_bytes())
        rc = hash_cli.main(
            ["--registry", str(tmp_path / "missing.json"), "--json", "lookup", str(img)]
        )
        capsys.readouterr()
        assert rc == hash_cli.EXIT_REGISTRY_UNAVAILABLE

    def test_lookup_invalid_registry_exits_registry_invalid(self, tmp_path, capsys):
        img = tmp_path / "img.png"
        img.write_bytes(_tiny_png_bytes())
        bad = tmp_path / "bad.json"
        bad.write_text("not json {")
        rc = hash_cli.main(
            ["--registry", str(bad), "--json", "lookup", str(img)]
        )
        capsys.readouterr()
        assert rc == hash_cli.EXIT_REGISTRY_INVALID

    def test_register_missing_file_error(self, tmp_path, capsys):
        rc = hash_cli.main(
            [
                "--registry", str(tmp_path / "r.json"),
                "register", str(tmp_path / "nope.png"),
                "--label", "unknown",
            ]
        )
        capsys.readouterr()
        assert rc == hash_cli.EXIT_ERROR

    def test_register_does_not_touch_image(self, tmp_path, capsys):
        img = tmp_path / "img.png"
        img.write_bytes(_tiny_png_bytes())
        before = img.stat()
        rc = hash_cli.main(
            [
                "--registry", str(tmp_path / "r.json"),
                "register", str(img),
                "--label", "camera_or_human",
            ]
        )
        capsys.readouterr()
        assert rc == hash_cli.EXIT_OK
        after = img.stat()
        assert before.st_mtime == after.st_mtime
        assert before.st_size == after.st_size
