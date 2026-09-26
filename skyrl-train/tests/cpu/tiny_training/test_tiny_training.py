"""End-to-end CPU training through the production entrypoints on a tiny policy."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.cpu.tiny_training.experiment import MAX_STALENESS_STEPS, TrainingMode, read_metrics

SKYRL_TRAIN_DIR = Path(__file__).parents[3]
NUM_STEPS = 3
# A run takes under a minute locally; the margin absorbs slower CI hosts, not hangs.
RUN_TIMEOUT_SECONDS = 150


@pytest.mark.parametrize("mode", list(TrainingMode))
def test_tiny_policy_trains_to_max_steps(tmp_path: Path, mode: TrainingMode):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.cpu.tiny_training.experiment",
            f"--mode={mode}",
            f"--steps={NUM_STEPS}",
            f"--root={tmp_path}",
        ],
        cwd=SKYRL_TRAIN_DIR,
        env={**os.environ, "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"},
        timeout=RUN_TIMEOUT_SECONDS,
        check=True,
    )

    steps = [record for record in read_metrics(tmp_path) if "policy/raw_grad_norm" in record]
    assert [record["trainer/global_step"] for record in steps] == list(range(1, NUM_STEPS + 1))
    assert all(record["policy/raw_grad_norm"] > 0 for record in steps)
    assert max(record["async/staleness_max"] for record in steps) <= MAX_STALENESS_STEPS[mode]
