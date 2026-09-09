import json
import subprocess
import sys

import pytest
import fsspec

from skyrl_train.entrypoints.startup_capture import TAIL_BYTES, capture


def test_pre_child_receipt_is_durable_before_command_and_failure_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "native-1")
    seen = []
    original = fsspec.core.url_to_fs

    def bounded_filesystem(url, **kwargs):
        seen.append(kwargs)
        return original(url, **kwargs)

    monkeypatch.setattr(fsspec.core, "url_to_fs", bounded_filesystem)
    receipt = tmp_path / "native-1/startup/bootstrap.json"
    command = [
        sys.executable,
        "-c",
        f"import json; assert json.load(open({str(receipt.with_name('bootstrap-before-child.json'))!r}))['state']=='before_child'; print('original failure'); raise SystemExit(7)",
    ]
    assert capture(command, str(tmp_path), "bootstrap", interval=0.01) == 7
    row = json.loads(receipt.read_text())
    assert row["exit_code"] == 7 and row["state"] == "finished"
    assert seen[0] == {
        "config_kwargs": {
            "connect_timeout": 5,
            "read_timeout": 10,
            "retries": {"max_attempts": 1},
            "s3": {"addressing_style": "virtual"},
        }
    }
    assert "original failure" in row["tail_utf8"]
    with pytest.raises(ValueError, match="duplicate"):
        capture(command, str(tmp_path), "bootstrap", interval=0.01)


def test_stdout_unicode_tail_is_bounded_and_serializable(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "native-2")
    command = [sys.executable, "-c", "print('雪' * 50000)"]
    assert capture(command, str(tmp_path), "unicode", interval=0.01) == 0
    payload = (tmp_path / "native-2/startup/unicode.json").read_bytes()
    row = json.loads(payload)
    assert row["tail_bytes"] == TAIL_BYTES and len(payload) < 1048576
    assert row["tail_utf8"].endswith("雪\n")


def test_actual_wrapper_captures_blocked_import_stack_before_runtime(tmp_path):
    module = tmp_path / "blocked_import.py"
    module.write_text("import time\ntime.sleep(0.15)\n")
    command = [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); from skyrl_train.entrypoints.startup_capture import run_module; run_module('blocked_import', [], stack_seconds=.02)",
        str(tmp_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "pre_import:blocked_import" in result.stdout
    assert "blocked_import.py" in result.stderr and "Timeout" in result.stderr


def test_invalid_native_identity_cannot_create_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "../bad")
    with pytest.raises(ValueError, match="identity"):
        capture([sys.executable, "-c", "pass"], str(tmp_path), "bootstrap")
    assert not tuple(tmp_path.iterdir())


def test_late_upload_failure_preserves_child_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "native-3")
    filesystem = fsspec.filesystem(
        "file",
        config_kwargs={
            "connect_timeout": 5,
            "read_timeout": 10,
            "retries": {"max_attempts": 1},
            "s3": {"addressing_style": "virtual"},
        },
    )
    original = filesystem.pipe
    calls = 0

    def fail_late(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 2:
            raise OSError("simulated upload interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(filesystem, "pipe", fail_late)
    assert capture([sys.executable, "-c", "raise SystemExit(9)"], str(tmp_path), "failure", interval=0.01) == 9
