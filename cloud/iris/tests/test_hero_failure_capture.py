import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

import fsspec
import pytest

from cloud.iris.task_runtime import sync_debug_artifacts
from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV, EnvVarScope
from scripts.hero_failure_capture import (
    LOSS_ARTIFACT,
    WORKER_ARTIFACT,
    check_artifacts,
    debug_environment,
    failure_marker,
)


def _retained_failure(tmp_path, monkeypatch):
    source = tmp_path / "debug"
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(source))
    manager = debug_environment("abort-test")
    manager.apply_to_process(EnvVarScope.TASK_RUNTIME)
    # A real CPU child writes into the worker-projected root; the runtime uploader
    # reads its own ambient root. A split between those roots loses these bytes.
    worker_code = (
        "import json,os\n"
        "from pathlib import Path\n"
        "root=Path(os.environ['SKYRL_DEBUG_ARTIFACT_DIR'])\n"
        "identity={'run_id':'abort-test','worker_id':'worker123','pid':456,'debug_root':str(root)}\n"
        "(root/'runs/worker-abort.json').write_text(json.dumps(identity))\n"
        "(root/'runs/worker-loss.json').write_text(json.dumps(identity))\n"
        "print(json.dumps(identity))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", worker_code],
        env={**os.environ, **manager.environment_for(EnvVarScope.RAY_WORKER)},
        check=True,
        capture_output=True,
        text=True,
    )
    identity = json.loads(result.stdout)
    output = f"memory://{tmp_path.name}"
    sync_debug_artifacts(f"{output}/rendezvous", "rank0-test", "driver exit_code=42 (head rank 0)")
    filesystem = fsspec.filesystem("memory")
    log = f"{output}/ray-logs/rank0-test/session_test/worker-worker123-01000000-456.err"
    filesystem.pipe(log, failure_marker(identity).encode() + b"\nFatal Python error: Aborted\n")
    return output, filesystem, log


@pytest.mark.parametrize("suffix", ["", "/"])
def test_acceptance_reads_failure_bytes_and_final_upload_receipt(tmp_path, monkeypatch, suffix):
    output, _, log = _retained_failure(tmp_path, monkeypatch)

    receipt = check_artifacts(output + suffix, "abort-test")

    assert receipt["identity"]["worker_id"] == "worker123"
    assert receipt["failure_logs"] == [fsspec.core.url_to_fs(log)[1]]
    assert receipt["manifest"]["copied_bytes"] > 0


@pytest.mark.parametrize("missing", [WORKER_ARTIFACT, LOSS_ARTIFACT, "sync-manifest.json", "stderr", "fatal-tail"])
def test_acceptance_rejects_lost_failure_evidence(tmp_path, monkeypatch, missing):
    output, filesystem, log = _retained_failure(tmp_path, monkeypatch)
    path = log if missing == "stderr" else f"{output}/rendezvous/debug_artifacts/rank0-test/{missing}"
    if missing == "fatal-tail":
        filesystem.pipe(log, b"HERO_CAPTURE_ABORT run=abort-test worker=worker123 pid=456\n")
    else:
        filesystem.rm(path)

    with pytest.raises((ValueError, FileNotFoundError)):
        check_artifacts(output, "abort-test")


def test_acceptance_rejects_artifacts_from_another_run(tmp_path, monkeypatch):
    output, _, _ = _retained_failure(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="identity"):
        check_artifacts(output, "another-run")


@pytest.mark.parametrize("root,interval", [(None, None), ("/tmp/explicit-debug", "17")])
def test_custom_launcher_sets_checkout_and_preserves_caller_environment(tmp_path, root, interval):
    # Substitute the external Python command to observe the real shell exports.
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os\n"
        "print(json.dumps({'checkout':os.environ['SKYRL_HOME'], 'root':os.environ.get('SKYRL_DEBUG_ARTIFACT_DIR'),"
        "'interval':os.environ.get('OT_AGENT_RAY_LOG_SYNC_INTERVAL_S')}))\n"
    )
    python.chmod(0o755)
    environment = {
        k: v for k, v in os.environ.items() if k not in (DEBUG_ARTIFACT_DIR_ENV, "OT_AGENT_RAY_LOG_SYNC_INTERVAL_S")
    }
    environment["PATH"] = f"{tmp_path}:{environment['PATH']}"
    environment["PYTHONPATH"] = str(Path.cwd())
    if root:
        environment[DEBUG_ARTIFACT_DIR_ENV] = root
        environment["OT_AGENT_RAY_LOG_SYNC_INTERVAL_S"] = interval

    result = subprocess.run(
        ["bash", "scripts/hero_failure_capture_task.sh"], env=environment, check=True, capture_output=True, text=True
    )
    wiring = json.loads(result.stdout)

    assert wiring["checkout"] == str(Path.cwd())
    assert wiring["root"] == root
    assert wiring["interval"] == interval


def test_signal_wrapper_waits_for_child_cleanup_before_exiting():
    child_code = (
        "import os,signal,sys,threading\n"
        "def finish(signum,frame):\n"
        " print('flushing',flush=True)\n"
        " assert sys.stdin.read(1)=='!'\n"
        " print('uploaded',flush=True)\n"
        " sys.exit(128+signum)\n"
        "signal.signal(signal.SIGTERM,finish)\n"
        "print(os.getpid(),flush=True)\n"
        "threading.Event().wait(30)\n"
    )
    runner_code = (
        "import sys\n"
        "from scripts.hero_failure_capture import supervise\n"
        f"sys.exit(supervise([sys.executable,'-c',{child_code!r}]))\n"
    )
    child_pid = None
    with subprocess.Popen(
        [sys.executable, "-c", runner_code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as runner:
        try:
            assert select.select([runner.stdout], [], [], 10)[0], "child did not become ready"
            child_pid = int(runner.stdout.readline())
            runner.send_signal(signal.SIGTERM)
            assert select.select([runner.stdout], [], [], 10)[0], "signal did not reach child"
            assert runner.stdout.readline().strip() == "flushing"
            assert runner.poll() is None
            output, _ = runner.communicate("!", timeout=10)
            assert output.strip() == "uploaded"
            assert runner.returncode == 128 + signal.SIGTERM
        finally:
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if runner.poll() is None:
                runner.kill()
