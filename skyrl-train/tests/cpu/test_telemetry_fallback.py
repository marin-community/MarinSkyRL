"""The optional-rigging fallback in ``skyrl_train/telemetry.py``."""

import os
import subprocess
import sys
from pathlib import Path

_SKYRL_TRAIN_ROOT = Path(__file__).resolve().parents[2]

_USE_INERT_BACKEND = """
import skyrl_train.inert_telemetry
import skyrl_train.telemetry as trainer_telemetry

assert trainer_telemetry.telemetry is skyrl_train.inert_telemetry
trainer_telemetry.record_policy_step(7)
with trainer_telemetry.critical_phase("train_step", 7):
    pass
"""


def test_rigging_without_telemetry_submodule_falls_back_to_inert(tmp_path: Path) -> None:
    (tmp_path / "rigging").mkdir()
    (tmp_path / "rigging" / "__init__.py").write_text("")

    # Shadowing the installed rigging takes a stub package on PYTHONPATH and a fresh
    # interpreter, because skyrl_train.telemetry binds its backend once, at import.
    result = subprocess.run(
        [sys.executable, "-c", _USE_INERT_BACKEND],
        cwd=_SKYRL_TRAIN_ROOT,
        env=os.environ | {"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_records_carry_the_step_they_describe(monkeypatch):
    """Generated work names the policy version that produced it; the rest name the step they ran in."""
    import skyrl_train.telemetry as trainer_telemetry

    recorded: list[tuple[str, dict]] = []

    class _Recorder:
        def __init__(self, name):
            self.name = name

        def add(self, _value, attributes):
            recorded.append((self.name, attributes))

        set = add
        record = add

    for name in (
        "work_completed",
        "progress_timestamp",
        "rollout_queue_depth",
        "rollout_capacity",
        "phase_duration",
        "policy_step",
    ):
        monkeypatch.setattr(trainer_telemetry, name, _Recorder(name))

    trainer_telemetry.record_generated_work([[1, 2], [3]], [True, True], 41, 39)
    trainer_telemetry.record_policy_step(41)
    trainer_telemetry.record_rollout_buffer(3, 8)
    with trainer_telemetry.critical_phase("train_step", 41):
        pass

    stamps = {
        (name, attributes.get("work_kind") or attributes.get("queue") or attributes.get("phase")): (
            attributes.get("step"),
            attributes.get("weights_step"),
        )
        for name, attributes in recorded
    }
    assert stamps == {
        # Both, so `step - weights_step` is the staleness with no time join.
        ("work_completed", "rollout"): ("41", "39"),
        ("work_completed", "sample"): ("41", "39"),
        ("work_completed", "generated_token"): ("41", "39"),
        ("work_completed", "policy_step"): ("41", None),
        ("phase_duration", "train_step"): ("41", None),
        # A live queue reading and a run constant; neither is about a step.
        ("rollout_queue_depth", "rollout_buffer"): (None, None),
        ("rollout_capacity", "rollout_buffer"): (None, None),
        # Liveness, read as a max over a window, and one name inside TRAINING_STATUS_NAMES.
        ("progress_timestamp", "rollout"): (None, None),
        ("progress_timestamp", "policy_step"): (None, None),
        # The gauge's value is the step.
        ("policy_step", None): (None, None),
    }
