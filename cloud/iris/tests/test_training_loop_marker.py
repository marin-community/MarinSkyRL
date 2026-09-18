from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.iris_backend import load_config_training_loop  # noqa: E402
from skyrl_train.telemetry import TelemetryConfig, _resources  # noqa: E402


@pytest.mark.parametrize(
    "body,expected", [("entrypoint: fully_async\n", "async"), ("entrypoint: standard\n", "sync"), ("", "sync")]
)
def test_training_loop_follows_the_configured_entrypoint(tmp_path, body, expected):
    config = tmp_path / "rl.yaml"
    config.write_text(body + "trainer: {}\n")
    assert load_config_training_loop(str(config)) == expected


def test_training_loop_lands_in_every_record_resource():
    resources = _resources(TelemetryConfig(run_id="run", execution_uid="x", training_loop="async"), "trainer")
    assert resources["training_loop"] == "async"
    assert "training_loop" not in _resources(TelemetryConfig(run_id="run", execution_uid="x"), "trainer")
