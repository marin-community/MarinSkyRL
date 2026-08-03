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
with trainer_telemetry.critical_phase("train_step"):
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
