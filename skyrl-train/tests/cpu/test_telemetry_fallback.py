"""Cover the inert-telemetry fallback against the rigging versions it has to tolerate.

Each case runs in a subprocess with a stub ``rigging`` ahead of the installed one on
``PYTHONPATH``, because the fallback is chosen once when ``skyrl_train.telemetry`` imports.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SKYRL_TRAIN_ROOT = Path(__file__).resolve().parents[2]

_HIDE_RIGGING = """
import sys


class _RiggingIsNotInstalled:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "rigging" or fullname.startswith("rigging."):
            raise ModuleNotFoundError(f"No module named '{fullname}'", name=fullname)
        return None


sys.meta_path.insert(0, _RiggingIsNotInstalled())
"""


def _run_with_stub_rigging(tmp_path: Path, body: str, *, telemetry_source: str = "", prelude: str = "") -> None:
    """Run ``body`` with a stub rigging whose telemetry submodule holds ``telemetry_source``, if any."""
    package = tmp_path / "rigging"
    package.mkdir()
    (package / "__init__.py").write_text("")
    if telemetry_source:
        (package / "telemetry.py").write_text(textwrap.dedent(telemetry_source))

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(prelude) + textwrap.dedent(body)],
        cwd=_SKYRL_TRAIN_ROOT,
        env=os.environ | {"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rigging_without_telemetry_submodule_leaves_the_trainer_signals_inert(tmp_path: Path) -> None:
    _run_with_stub_rigging(
        tmp_path,
        """
        import skyrl_train.inert_telemetry
        import skyrl_train.telemetry as trainer_telemetry

        assert trainer_telemetry.telemetry is skyrl_train.inert_telemetry

        trainer_telemetry.record_policy_step(7)
        trainer_telemetry.record_generated_work([[1, 2], [3]], [True, False])
        trainer_telemetry.record_rollout_buffer(3, 16)
        with trainer_telemetry.critical_phase("train_step"):
            pass

        collector = object()
        with trainer_telemetry.process_telemetry(trainer_telemetry.TRAINER_ROLE) as owner:
            assert owner.collector_or_inert(collector) is not collector
        """,
    )


def test_absent_rigging_leaves_the_trainer_signals_inert(tmp_path: Path) -> None:
    _run_with_stub_rigging(
        tmp_path,
        """
        import skyrl_train.inert_telemetry
        import skyrl_train.telemetry as trainer_telemetry

        assert trainer_telemetry.telemetry is skyrl_train.inert_telemetry
        trainer_telemetry.record_policy_step(7)
        """,
        prelude=_HIDE_RIGGING,
    )


def test_import_error_from_inside_rigging_telemetry_propagates(tmp_path: Path) -> None:
    _run_with_stub_rigging(
        tmp_path,
        """
        try:
            import skyrl_train.telemetry
        except ImportError as error:
            assert error.name == "collections", error.name
        else:
            raise AssertionError("an ImportError raised inside rigging.telemetry must not be swallowed")
        """,
        telemetry_source="from collections import this_symbol_does_not_exist\n",
    )


def test_rigging_without_telemetry_submodule_fails_ray_metrics_with_import_error_named_rigging(
    tmp_path: Path,
) -> None:
    """``start_rl_iris_controller`` keys its fallback on this exception shape from another package."""
    pytest.importorskip("prometheus_client")
    _run_with_stub_rigging(
        tmp_path,
        """
        try:
            import skyrl_train.ray_metrics
        except ImportError as error:
            assert error.name == "rigging", error.name
        else:
            raise AssertionError("ray_metrics cannot import telemetry from a rigging that lacks it")
        """,
    )
