from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from marinskyrl.export_completion import ExportReceipt, verify_export_receipt


def _write_export(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )


def test_export_receipt_verifies_source_binding_and_all_shards(tmp_path: Path) -> None:
    export_path = tmp_path / "exports" / "global_step_12" / "policy"
    receipt_path = tmp_path / "receipts" / "attempt-1.json"
    _write_export(export_path)
    receipt_path.parent.mkdir()
    receipt_path.write_text(
        json.dumps(
            ExportReceipt(
                request_fingerprint="abc123",
                attempt_id="attempt-1",
                export_path=str(export_path),
                global_step=12,
            ).to_dict()
        )
    )

    receipt = verify_export_receipt(str(receipt_path), "abc123", str(export_path), 12)

    assert receipt.attempt_id == "attempt-1"
    (export_path / "model-00002-of-00002.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="missing 1 nonempty safetensors shard"):
        verify_export_receipt(str(receipt_path), "abc123", str(export_path), 12)


def test_export_receipt_rejects_wrong_request_without_adopting_files(tmp_path: Path) -> None:
    export_path = tmp_path / "policy"
    receipt_path = tmp_path / "receipt.json"
    _write_export(export_path)
    receipt_path.write_text(
        json.dumps(
            ExportReceipt(
                request_fingerprint="old-request",
                attempt_id="attempt-1",
                export_path=str(export_path),
                global_step=4,
            ).to_dict()
        )
    )

    with pytest.raises(ValueError, match="does not match this request"):
        verify_export_receipt(str(receipt_path), "new-request", str(export_path), 4)


def test_export_completion_module_does_not_import_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import marinskyrl.export_completion; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
