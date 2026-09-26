"""End-to-end CPU training through the production entrypoints on a tiny policy."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from finestore.layout import BlobTables
from finestore.reader import ReadView

from skyrl_train.rollouts.payloads import ROLLOUT_OBJECT_PREFIX
from tests.cpu.tiny_training.experiment import (
    FINESTORE_ARCHIVE,
    MAX_STALENESS_STEPS,
    N_SAMPLES_PER_PROMPT,
    TRAIN_BATCH_SIZE,
    RolloutShape,
    TrainingMode,
    read_metrics,
    sampling_kind,
)
from tests.cpu.tiny_training.tiny_model import CURRICULUM_BINS

SKYRL_TRAIN_DIR = Path(__file__).parents[3]
NUM_STEPS = 3
# A run takes under a minute locally; the margin absorbs slower CI hosts, not hangs.
RUN_TIMEOUT_SECONDS = 150


@pytest.mark.parametrize("shape", list(RolloutShape))
@pytest.mark.parametrize("mode", list(TrainingMode))
def test_tiny_policy_trains_to_max_steps(tmp_path: Path, mode: TrainingMode, shape: RolloutShape):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.cpu.tiny_training.experiment",
            f"--mode={mode}",
            f"--shape={shape}",
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
    # A batch whose samples all score alike, which happens by chance, leaves no advantage to train on.
    assert any(record["policy/raw_grad_norm"] > 0 for record in steps)
    assert max(record["async/staleness_max"] for record in steps) <= MAX_STALENESS_STEPS[mode]
    if shape is RolloutShape.STEP_WISE:
        # A trajectory that answers wrongly on its first turn trains one sample per turn.
        trajectories = NUM_STEPS * TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT
        assert sum(record["consumed/sequences"] for record in steps) > trajectories
    if sampling_kind(mode, shape) is not None:
        # Without dynamic sampling, the groups each step judged are exactly its batch.
        for record in steps:
            assert sum(record[f"curriculum/{name}/groups"] for name in CURRICULUM_BINS) == TRAIN_BATCH_SIZE
    if FINESTORE_ARCHIVE[mode]:
        # Every trained group was read back from the archive, which also keeps groups generated but not trained.
        names = ReadView(str(tmp_path / "rollouts")).keys(BlobTables.DESCRIPTORS)
        assert sum(name.startswith(ROLLOUT_OBJECT_PREFIX) for (name,) in names) >= NUM_STEPS * TRAIN_BATCH_SIZE
