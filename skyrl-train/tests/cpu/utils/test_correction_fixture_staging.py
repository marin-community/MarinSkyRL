"""Actual streaming-copy identity checks for the bounded native fixture stage."""

import hashlib
import io

import pytest

from tests.gpu.diagnostics.stage_qwen_correction_fixture import copy_verified


@pytest.mark.parametrize("change", ["none", "wrong_hash", "truncated", "oversized"])
def test_streamed_artifact_requires_digest_and_size(tmp_path, change):
    payload = b"actual immutable fixture bytes"
    digest = hashlib.sha256(payload).hexdigest()
    incoming = payload[:-1] if change == "truncated" else payload + b"extra" if change == "oversized" else payload
    target = tmp_path / "model.safetensors"
    expected = "0" * 64 if change == "wrong_hash" else digest
    if change == "none":
        receipt = copy_verified(io.BytesIO(incoming), target, expected, expected_size=len(payload))
        assert receipt == {"sha256": digest, "bytes": len(payload)} and target.read_bytes() == payload
    else:
        with pytest.raises(ValueError):
            copy_verified(io.BytesIO(incoming), target, expected, expected_size=len(payload))
        assert not target.exists()
    assert not target.with_suffix(".safetensors.partial").exists()
