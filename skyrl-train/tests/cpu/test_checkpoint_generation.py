import json
from pathlib import Path

import pytest

from marinskyrl.checkpoint_paths import extract_step_from_path
from skyrl_train.checkpoint_generation import (
    COMMIT_FILENAME,
    MANIFEST_FILENAME,
    commit_attempt,
    new_attempt_path,
    resolve_checkpoint_payload,
)
from skyrl_train.checkpoint_listing import list_committed_checkpoint_dirs
from skyrl_train.io import io
from skyrl_train.utils.trainer_utils import validate_consistency_for_latest_checkpoint


def _attempt(step_path: Path, payload: bytes = b"weights") -> Path:
    attempt = Path(new_attempt_path(str(step_path)))
    (attempt / "policy").mkdir(parents=True)
    (attempt / "policy" / "rank_0.pt").write_bytes(payload)
    (attempt / "trainer_state.pt").write_bytes(b"trainer")
    (attempt / "data.pt").write_bytes(b"dataloader")
    return attempt


_REQUIRED = {"policy/rank_0.pt", "trainer_state.pt", "data.pt"}


def test_uncommitted_attempt_is_not_loadable(tmp_path):
    step_path = tmp_path / "global_step_1"
    attempt = _attempt(step_path)
    assert extract_step_from_path(str(attempt / "policy")) == 1

    with pytest.raises(FileNotFoundError, match="No committed checkpoint"):
        resolve_checkpoint_payload(str(step_path))


@pytest.mark.parametrize("read_failure", [False, True])
def test_partial_later_step_does_not_displace_latest_committed_generation(tmp_path, monkeypatch, read_failure):
    committed_step = tmp_path / "global_step_1"
    completed = _attempt(committed_step)
    commit_attempt(str(committed_step), str(completed), required_files=_REQUIRED)
    _attempt(tmp_path / "global_step_2")

    if read_failure:

        def unavailable(path):
            raise OSError("injected object-store read failure")

        monkeypatch.setattr(io, "read_bytes", unavailable)
        with pytest.raises(OSError, match="injected object-store read failure"):
            list_committed_checkpoint_dirs(str(tmp_path))
    else:
        assert list_committed_checkpoint_dirs(str(tmp_path)) == ["global_step_1"]


def test_committed_but_unadvertised_later_step_does_not_block_previous_resume(tmp_path):
    previous_step = tmp_path / "global_step_1"
    commit_attempt(str(previous_step), str(_attempt(previous_step)), required_files=_REQUIRED)
    unadvertised_step = tmp_path / "global_step_3"
    commit_attempt(str(unadvertised_step), str(_attempt(unadvertised_step)), required_files=_REQUIRED)
    latest = tmp_path / "latest_ckpt_global_step.txt"
    latest.write_text("1")

    validate_consistency_for_latest_checkpoint(str(tmp_path), 1, str(previous_step), str(latest), save_interval=1)
    assert resolve_checkpoint_payload(str(previous_step), verify_files=True)


def test_commit_resolves_exact_attempt_and_verifies_inventory(tmp_path):
    step_path = tmp_path / "global_step_1"
    attempt = _attempt(step_path)

    commit_attempt(str(step_path), str(attempt), required_files=_REQUIRED)

    assert resolve_checkpoint_payload(str(step_path), verify_files=True) == str(attempt)
    (attempt / "policy" / "rank_0.pt").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="size changed"):
        resolve_checkpoint_payload(str(step_path), verify_files=True)


@pytest.mark.parametrize("failure_boundary", ["missing_rank", "manifest_write", "commit_write"])
def test_failed_replacement_preserves_previous_generation(tmp_path, monkeypatch, failure_boundary):
    step_path = tmp_path / "global_step_1"
    previous = _attempt(step_path, b"old weights")
    commit_attempt(str(step_path), str(previous), required_files=_REQUIRED)
    previous_commit = (step_path / COMMIT_FILENAME).read_bytes()
    candidate = _attempt(step_path, b"new weights")
    if failure_boundary == "missing_rank":
        (candidate / "policy" / "rank_0.pt").unlink()
    else:
        real_write = io.write_bytes_atomic
        target = MANIFEST_FILENAME if failure_boundary == "manifest_write" else COMMIT_FILENAME

        def fail_at_target(path, payload):
            if path.endswith(target):
                raise OSError(f"injected {failure_boundary} failure")
            real_write(path, payload)

        monkeypatch.setattr(io, "write_bytes_atomic", fail_at_target)

    if failure_boundary == "missing_rank":
        with pytest.raises(RuntimeError, match="missing"):
            commit_attempt(str(step_path), str(candidate), required_files=_REQUIRED)
    else:
        with pytest.raises(OSError, match="injected"):
            commit_attempt(str(step_path), str(candidate), required_files=_REQUIRED)

    assert (step_path / COMMIT_FILENAME).read_bytes() == previous_commit
    assert resolve_checkpoint_payload(str(step_path), verify_files=True) == str(previous)


def test_legacy_checkpoint_remains_readable_and_corrupt_commit_does_not_fall_back(tmp_path):
    step_path = tmp_path / "global_step_1"
    step_path.mkdir()
    (step_path / "trainer_state.pt").write_bytes(b"legacy")
    assert resolve_checkpoint_payload(str(step_path)) == str(step_path)

    (step_path / COMMIT_FILENAME).write_text(json.dumps({"schema_version": 1, "step": 1, "attempt_id": "bad"}))
    with pytest.raises(ValueError, match="attempt ID"):
        resolve_checkpoint_payload(str(step_path))
