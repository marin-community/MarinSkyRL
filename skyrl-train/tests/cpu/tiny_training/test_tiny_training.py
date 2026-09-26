"""End-to-end CPU training through the production entrypoints on a tiny policy."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
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
RESUMED_STEP = 2
# A run takes under a minute locally; the margin absorbs slower CI hosts, not hangs.
RUN_TIMEOUT_SECONDS = 150


def _train(root: Path, mode: TrainingMode, shape: RolloutShape, *, steps: int, checkpoint_interval: int = -1) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.cpu.tiny_training.experiment",
            f"--mode={mode}",
            f"--shape={shape}",
            f"--steps={steps}",
            f"--checkpoint-interval={checkpoint_interval}",
            f"--root={root}",
        ],
        cwd=SKYRL_TRAIN_DIR,
        env={**os.environ, "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"},
        timeout=RUN_TIMEOUT_SECONDS,
        check=True,
    )


def _trained_steps(root: Path) -> list[dict]:
    return [record for record in read_metrics(root) if "policy/raw_grad_norm" in record]


@pytest.mark.parametrize("shape", list(RolloutShape))
@pytest.mark.parametrize("mode", list(TrainingMode))
def test_tiny_policy_trains_to_max_steps(tmp_path: Path, mode: TrainingMode, shape: RolloutShape):
    _train(tmp_path, mode, shape, steps=NUM_STEPS)

    steps = _trained_steps(tmp_path)
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


def test_async_training_resumes_with_committed_groups(tmp_path: Path):
    _train(tmp_path, TrainingMode.ASYNC, RolloutShape.SINGLE_TURN, steps=RESUMED_STEP, checkpoint_interval=1)
    # Generation runs ahead of training, so the checkpoint holds groups committed for the next batch.
    state = torch.load(tmp_path / "ckpts" / f"global_step_{RESUMED_STEP}" / "data.pt", weights_only=False)
    assert state.ready

    _train(tmp_path, TrainingMode.ASYNC, RolloutShape.SINGLE_TURN, steps=NUM_STEPS, checkpoint_interval=1)

    assert [record["trainer/global_step"] for record in _trained_steps(tmp_path)] == list(range(1, NUM_STEPS + 1))
