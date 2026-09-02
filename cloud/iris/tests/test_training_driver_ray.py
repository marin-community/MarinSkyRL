import os
import subprocess
import sys

import pytest


_PROBE = """
from cloud.iris.training_driver import LocalRLConfig, LocalRLRunner

runner = LocalRLRunner(LocalRLConfig(
    rl_config_path="unused.yaml",
    job_name="test-job",
    model_path="Qwen/Qwen3-0.6B",
    num_nodes={num_nodes},
))
raise SystemExit(runner._run_skyrl("site", []))
"""


@pytest.mark.parametrize(("num_nodes", "expected_returncode"), [(1, 0), (2, 1)])
def test_training_driver_only_requires_external_ray_for_multiple_nodes(num_nodes: int, expected_returncode: int) -> None:
    environment = os.environ.copy()
    environment.pop("RAY_ADDRESS", None)

    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(num_nodes=num_nodes)],
        env=environment,
        check=False,
    )

    assert result.returncode == expected_returncode
