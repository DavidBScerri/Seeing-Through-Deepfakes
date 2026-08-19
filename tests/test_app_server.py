"""
Tests for the web-app request plumbing
(src/genai_detection/integration_pipeline/app.py) and the YuNet model
download/verification (src/deepfake_detection/deepfake_classifier.py).

No HTTP server or model is started — these exercise the pure helpers.
"""

import hashlib
import os
import urllib.request

import pytest

from src.genai_detection.integration_pipeline.app import parse_multipart_form, _temp_suffix_for
import src.deepfake_detection.deepfake_classifier as dc


def _multipart_body(boundary, file_bytes, filename='photo.png', params=None):
    parts = [
        f"--{boundary}\r\n".encode()
        + f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
        + b"Content-Type: application/octet-stream\r\n\r\n"
        + file_bytes
        + b"\r\n"
    ]
    for name, value in (params or {}).items():
        parts.append(
            f"--{boundary}\r\n".encode()
            + f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            + value.encode()
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts)


class TestParseMultipartForm:
    def test_binary_payload_preserved_exactly(self):
        boundary = "----WebKitFormBoundaryAAA111"
        # CR/LF pairs, NULs, and boundary-like bytes inside the payload
        file_bytes = b"\x89PNG\r\n\x1a\n--fakeboundary\r\n" + bytes(range(256)) * 8
        body = _multipart_body(boundary, file_bytes, params={"mode": "fast"})
        data, filename, params = parse_multipart_form(body, f"multipart/form-data; boundary={boundary}")
        assert data == file_bytes
        assert filename == "photo.png"
        assert params == {"mode": "fast"}

    def test_quoted_boundary_and_extra_content_type_params(self):
        boundary = "xYzBoundary42"
        file_bytes = b"payload-bytes"
        body = _multipart_body(boundary, file_bytes)
        ct = f'multipart/form-data; charset=utf-8; boundary="{boundary}"'
        data, filename, _ = parse_multipart_form(body, ct)
        assert data == file_bytes

    def test_empty_filename_still_counts_as_file_part(self):
        boundary = "bnd"
        file_bytes = b"\x00\x01binary"
        body = _multipart_body(boundary, file_bytes, filename="")
        data, filename, params = parse_multipart_form(body, f"multipart/form-data; boundary={boundary}")
        assert data == file_bytes
        assert filename == ""
        assert params == {}

    def test_missing_boundary_returns_no_file(self):
        data, filename, params = parse_multipart_form(b"whatever", "multipart/form-data")
        assert data is None
        assert filename is None
        assert params == {}

    def test_garbage_body_returns_no_file(self):
        data, _, _ = parse_multipart_form(b"\x00\xffnot multipart at all", "multipart/form-data; boundary=nope")
        assert data is None


class TestTempSuffix:
    def test_known_image_suffix_is_kept_lowercased(self):
        assert _temp_suffix_for("photo.JPEG") == ".jpeg"
        assert _temp_suffix_for("img.webp") == ".webp"

    def test_unknown_or_missing_suffix_falls_back_to_png(self):
        assert _temp_suffix_for("evil.exe") == ".png"
        assert _temp_suffix_for("noext") == ".png"
        assert _temp_suffix_for("") == ".png"
        assert _temp_suffix_for(None) == ".png"


class TestYunetModelPath:
    def test_download_verifies_checksum(self, tmp_path, monkeypatch):
        content = b"pretend-onnx-model"
        monkeypatch.setattr(dc, "_YUNET_SHA256", hashlib.sha256(content).hexdigest())
        monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dst: open(dst, "wb").write(content))
        model_path = str(tmp_path / "yunet.onnx")
        assert dc._get_yunet_model_path(model_path) == model_path
        assert open(model_path, "rb").read() == content

    def test_checksum_mismatch_blocks_installation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dst: open(dst, "wb").write(b"tampered-bytes"))
        model_path = str(tmp_path / "yunet.onnx")
        with pytest.raises(RuntimeError, match="checksum"):
            dc._get_yunet_model_path(model_path)
        assert not os.path.exists(model_path)
        assert not os.path.exists(model_path + ".download")

    def test_existing_file_mismatch_warns_but_is_not_deleted(self, tmp_path, capsys):
        model_path = tmp_path / "yunet.onnx"
        model_path.write_bytes(b"locally-modified-model")
        result = dc._get_yunet_model_path(str(model_path))
        assert result == str(model_path)
        assert model_path.read_bytes() == b"locally-modified-model"
        assert "checksum" in capsys.readouterr().out.lower()

    def test_existing_valid_file_passes_silently(self, tmp_path, monkeypatch, capsys):
        content = b"valid-model"
        monkeypatch.setattr(dc, "_YUNET_SHA256", hashlib.sha256(content).hexdigest())
        model_path = tmp_path / "yunet.onnx"
        model_path.write_bytes(content)
        assert dc._get_yunet_model_path(str(model_path)) == str(model_path)
        assert "warning" not in capsys.readouterr().out.lower()
