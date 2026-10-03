import json

import fsspec
import pytest

from cloud.iris.task_runtime import sync_debug_artifacts
from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV, EnvVarScope
from scripts.hero_failure_capture import WORKER_ARTIFACT, check_artifacts, debug_environment, failure_marker


@pytest.mark.parametrize("damage", [None, "artifact", "manifest", "stderr", "fatal-tail", "wrong-run"])
def test_retained_failure_acceptance(tmp_path, monkeypatch, damage):
    source = tmp_path / "debug"
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(source))
    debug_environment("abort-test").apply_to_process(EnvVarScope.TASK_RUNTIME)
    identity = {"run_id": "abort-test", "worker_id": "worker123", "pid": 456, "debug_root": str(source)}
    (source / WORKER_ARTIFACT).write_text(json.dumps(identity))
    output = f"memory://{tmp_path.name}"
    sync_debug_artifacts(f"{output}/rendezvous", "node-0", "driver exit_code=42 (head rank 0)")
    filesystem = fsspec.filesystem("memory")
    artifact = f"{output}/rendezvous/debug_artifacts/node-0/{WORKER_ARTIFACT}"
    manifest = f"{output}/rendezvous/debug_artifacts/node-0/sync-manifest.json"
    log = f"{output}/ray-logs/node-0/session_test/worker-worker123-01000000-456.err"
    marker = failure_marker(identity).encode()
    filesystem.pipe(log, marker + b"\nFatal Python error: Aborted\n")
    if damage in {"artifact", "manifest", "stderr"}:
        filesystem.rm({"artifact": artifact, "manifest": manifest, "stderr": log}[damage])
    elif damage == "fatal-tail":
        filesystem.pipe(log, marker)
    run_id = "other-run" if damage == "wrong-run" else "abort-test"
    if damage:
        with pytest.raises((ValueError, FileNotFoundError)):
            check_artifacts(output, run_id)
    else:
        receipt = check_artifacts(output, run_id)
        assert receipt["identity"] == identity
        assert receipt["stderr"] == fsspec.core.url_to_fs(log)[1]
