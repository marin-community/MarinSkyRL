"""Covers the ImportError guard in ``start_rl_iris_controller.py``; delete both together."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_rigging_without_telemetry_submodule_leaves_ray_metric_forwarding_disabled(tmp_path: Path) -> None:
    (tmp_path / "rigging").mkdir()
    (tmp_path / "rigging" / "__init__.py").write_text("")
    (tmp_path / "skyrl_train").mkdir()
    (tmp_path / "skyrl_train" / "__init__.py").write_text("")
    (tmp_path / "skyrl_train" / "ray_metrics.py").write_text("from rigging import telemetry as rigging_telemetry\n")

    # Stub packages on PYTHONPATH and a fresh interpreter, because the controller resolves
    # ray_metrics once, at import.
    script = textwrap.dedent(
        """
        from cloud.iris import start_rl_iris_controller

        with start_rl_iris_controller.ray_metrics_telemetry("10.0.0.1", 8080):
            pass
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=os.environ | {"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr
