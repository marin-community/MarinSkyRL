import json
from pathlib import Path

import fsspec

from cloud.iris.task_runtime import sync_debug_artifacts
from skyrl_train.env_vars import DEBUG_ARTIFACT_DIR_ENV


def test_debug_sync_persists_files_and_complete_manifest(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "debug"
    (artifact_root / "flight_recorder").mkdir(parents=True)
    (artifact_root / "processes").mkdir()
    (artifact_root / "flight_recorder" / "nccl_fr_rank_0").write_bytes(b"flight")
    (artifact_root / "processes" / "rank0.json").write_text('{"rank": 0}\n')
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(artifact_root))

    sync_debug_artifacts("memory://debug-contract", "node-0", "test")

    filesystem = fsspec.filesystem("memory")
    base = "/debug-contract/debug_artifacts/node-0"
    assert filesystem.cat(f"{base}/flight_recorder/nccl_fr_rank_0") == b"flight"
    manifest = json.loads(filesystem.cat(f"{base}/sync-manifest.json"))
    assert {item["path"] for item in manifest["copied"]} == {
        "flight_recorder/nccl_fr_rank_0",
        "processes/rank0.json",
    }
    assert manifest["skipped"] == []


def test_debug_sync_records_files_rejected_by_budget(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "debug"
    artifact_root.mkdir()
    (artifact_root / "oversized.bin").write_bytes(b"12345")
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(artifact_root))
    monkeypatch.setattr("cloud.iris.task_runtime.DEBUG_SYNC_MAX_FILE_BYTES", 4)

    sync_debug_artifacts("memory://debug-budget", "node-1", "test")

    filesystem = fsspec.filesystem("memory")
    manifest = json.loads(filesystem.cat("/debug-budget/debug_artifacts/node-1/sync-manifest.json"))
    assert manifest["copied"] == []
    assert manifest["skipped"] == [{"bytes": 5, "path": "oversized.bin", "reason": "budget"}]
