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

    recorded: list[dict] = []

    class _Recorder:
        def add(self, _value, attributes):
            recorded.append(attributes)

        def set(self, _value, attributes):
            recorded.append(attributes)

        def record(self, _value, attributes):
            recorded.append(attributes)

    for name in (
        "work_completed",
        "progress_timestamp",
        "rollout_queue_depth",
        "rollout_capacity",
        "phase_duration",
        "policy_step",
    ):
        monkeypatch.setattr(trainer_telemetry, name, _Recorder())

    trainer_telemetry.record_generated_work([[1, 2], [3]], [True, True], 39)
    trainer_telemetry.record_policy_step(41)
    trainer_telemetry.record_rollout_buffer(3, 8)
    with trainer_telemetry.critical_phase("train_step", 41):
        pass

    assert recorded, "no telemetry records were emitted"
    stamped = [attributes for attributes in recorded if "step" in attributes or "weights_step" in attributes]
    assert {attributes.get("step") for attributes in stamped} == {"41", None}
    assert {attributes.get("weights_step") for attributes in stamped} == {"39", None}
    # A record carries one of the two names, never both, so neither join double-counts.
    assert not [a for a in stamped if "step" in a and "weights_step" in a]
