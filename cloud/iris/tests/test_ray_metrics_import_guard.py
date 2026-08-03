"""Cover the controller's optional-import guard around ``skyrl_train.ray_metrics``.

Each case runs in a subprocess against stub ``rigging`` and ``skyrl_train`` packages, so the
controller sees the same import failures a GPU image with an older rigging produces.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_RAY_METRICS_HEAD = "from rigging import telemetry as rigging_telemetry\n"


def _import_controller_with_stub_ray_metrics(tmp_path: Path, ray_metrics_source: str, body: str) -> None:
    (tmp_path / "rigging").mkdir()
    (tmp_path / "rigging" / "__init__.py").write_text("")
    (tmp_path / "skyrl_train").mkdir()
    (tmp_path / "skyrl_train" / "__init__.py").write_text("")
    (tmp_path / "skyrl_train" / "ray_metrics.py").write_text(ray_metrics_source)

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        cwd=_REPO_ROOT,
        env=os.environ | {"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rigging_without_telemetry_submodule_leaves_ray_metric_forwarding_disabled(tmp_path: Path) -> None:
    _import_controller_with_stub_ray_metrics(
        tmp_path,
        _RAY_METRICS_HEAD,
        """
        from cloud.iris import start_rl_iris_controller

        with start_rl_iris_controller.ray_metrics_telemetry("10.0.0.1", 8080):
            pass
        """,
    )


def test_unrelated_import_error_in_ray_metrics_propagates(tmp_path: Path) -> None:
    _import_controller_with_stub_ray_metrics(
        tmp_path,
        "from collections import this_symbol_does_not_exist\n",
        """
        try:
            from cloud.iris import start_rl_iris_controller
        except ImportError as error:
            assert error.name == "collections", error.name
        else:
            raise AssertionError("an unrelated ImportError from ray_metrics must not be swallowed")
        """,
    )
