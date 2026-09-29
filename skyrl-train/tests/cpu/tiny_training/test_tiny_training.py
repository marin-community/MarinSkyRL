"""End-to-end CPU training through the production entrypoints on a tiny policy."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from marinskyrl.checkpoint_paths import POLICY_CHECKPOINT_SUBDIRECTORY
from skyrl_train.rollouts.payloads import ROLLOUT_OBJECT_SUFFIX
from skyrl_train.training_batch import TrainingInputBatch
from tests.cpu.tiny_training.cpu_backend import CHECKPOINT_FILE_TEMPLATE
from tests.cpu.tiny_training.experiment import (
    MAX_STALENESS_STEPS,
    N_SAMPLES_PER_PROMPT,
    OBJECT_STORE_PAYLOADS,
    TRAIN_BATCH_SIZE,
    RolloutShape,
    TrainingMode,
    read_metrics,
    sampling_kind,
)
from tests.cpu.tiny_training.tiny_model import CURRICULUM_BINS

SKYRL_TRAIN_DIR = Path(__file__).parents[3]
NUM_STEPS = 3
RESUMED_STEP = 1
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
            *(["--dump-data-batch"] if mode is TrainingMode.SYNC else []),
        ],
        cwd=SKYRL_TRAIN_DIR,
        env={**os.environ, "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"},
        timeout=RUN_TIMEOUT_SECONDS,
        check=True,
    )


def _trained_steps(root: Path) -> list[dict]:
    return [record for record in read_metrics(root) if "policy/raw_grad_norm" in record]


def test_one_step_is_independent_of_micro_batch_size(tmp_path: Path):
    roots = [tmp_path / "micro_8", tmp_path / "micro_4"]
    for root, micro_batch_size in zip(roots, [8, 4], strict=True):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.cpu.tiny_training.fixed_batch",
                f"--root={root}",
                f"--micro-batch-size={micro_batch_size}",
            ],
            cwd=SKYRL_TRAIN_DIR,
            env={**os.environ, "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"},
            timeout=RUN_TIMEOUT_SECONDS,
            check=True,
        )
    batches = [
        TrainingInputBatch().load(root / "exports/dumped_data/global_step_1_training_input.pkl") for root in roots
    ]
    assert batches[0].keys() == batches[1].keys()
    for key, tensor in batches[0].items():
        if tensor is None:
            assert batches[1][key] is None
        else:
            torch.testing.assert_close(
                tensor, batches[1][key], rtol=0, atol=0, msg=f"nondeterministic rollouts in {key}"
            )
    lengths = batches[0]["loss_mask"].sum(-1)
    assert lengths.min() < lengths.max()
    for rank in range(2):
        states = [
            torch.load(
                root
                / "ckpts/global_step_1"
                / POLICY_CHECKPOINT_SUBDIRECTORY
                / CHECKPOINT_FILE_TEMPLATE.format(rank=rank),
                weights_only=False,
            )
            for root in roots
        ]
        for key, tensor in states[0]["model"].items():
            torch.testing.assert_close(tensor, states[1]["model"][key], rtol=1e-5, atol=1e-6, msg=key)


@pytest.mark.parametrize("shape", list(RolloutShape))
@pytest.mark.parametrize("mode", list(TrainingMode))
def test_tiny_policy_trains_to_max_steps(tmp_path: Path, mode: TrainingMode, shape: RolloutShape):
    _train(tmp_path, mode, shape, steps=NUM_STEPS)

    steps = _trained_steps(tmp_path)
    assert [record["trainer/global_step"] for record in steps] == list(range(1, NUM_STEPS + 1))
    # A batch whose samples all score alike, which happens by chance, leaves no advantage to train on.
    assert any(record["policy/raw_grad_norm"] > 0 for record in steps)
    assert max(record["async/staleness_max"] for record in steps) <= MAX_STALENESS_STEPS[mode]
    if mode is TrainingMode.SYNC:
        for step in range(1, NUM_STEPS + 1):
            batch = TrainingInputBatch().load(tmp_path / f"exports/dumped_data/global_step_{step}_training_input.pkl")
            eligible = batch["loss_mask"].bool()
            expected = torch.zeros_like(batch["action_log_probs"], dtype=torch.float32)
            expected[eligible] = (
                (batch["action_log_probs"][eligible].float() - batch["rollout_logprobs"][eligible].float())
                .clamp(-20, 20)
                .exp()
                .clamp(max=2)
            )
            torch.testing.assert_close(batch["correction_weights"], expected)
            assert steps[step - 1]["policy/correction/weight_mean"] == pytest.approx(
                expected.sum().item() / eligible.sum().item()
            )
    if shape is RolloutShape.STEP_WISE:
        # A trajectory that answers wrongly on its first turn trains one sample per turn.
        trajectories = NUM_STEPS * TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT
        assert sum(record["consumed/sequences"] for record in steps) > trajectories
    if sampling_kind(mode, shape) is not None:
        # Without dynamic sampling, the groups each step judged are exactly its batch.
        for record in steps:
            assert sum(record[f"curriculum/{name}/groups"] for name in CURRICULUM_BINS) == TRAIN_BATCH_SIZE
    if OBJECT_STORE_PAYLOADS[mode]:
        # Every trained group was read back from its object, and the objects stay for groups generated but not
        # trained.
        objects = list((tmp_path / "rollouts").glob(f"*{ROLLOUT_OBJECT_SUFFIX}"))
        assert len(objects) >= NUM_STEPS * TRAIN_BATCH_SIZE


def test_async_training_resumes_with_committed_groups(tmp_path: Path):
    _train(tmp_path, TrainingMode.ASYNC, RolloutShape.SINGLE_TURN, steps=RESUMED_STEP, checkpoint_interval=1)
    # The first two batches' leases open together, so the second batch's groups commit while the first trains
    # and its checkpoint holds them. A later step's leases open only when its batch is taken, and whether they
    # commit before that step's checkpoint is a race.
    state = torch.load(tmp_path / "ckpts" / f"global_step_{RESUMED_STEP}" / "data.pt", weights_only=False)
    assert state.ready

    _train(tmp_path, TrainingMode.ASYNC, RolloutShape.SINGLE_TURN, steps=NUM_STEPS, checkpoint_interval=1)

    assert [record["trainer/global_step"] for record in _trained_steps(tmp_path)] == list(range(1, NUM_STEPS + 1))
