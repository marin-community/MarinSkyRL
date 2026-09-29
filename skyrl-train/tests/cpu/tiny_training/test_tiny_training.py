"""End-to-end CPU training through the production entrypoints on a tiny policy."""

import multiprocessing
from multiprocessing.context import ForkServerContext
from pathlib import Path

import pytest
import torch

from marinskyrl.checkpoint_paths import POLICY_CHECKPOINT_SUBDIRECTORY
from skyrl_train.training_batch import TrainingInputBatch
from tests.cpu.tiny_training.cpu_backend import CHECKPOINT_FILE_TEMPLATE
from tests.cpu.tiny_training.fixed_batch import run_fixed_update
from skyrl_train.rollouts.payloads import ROLLOUT_OBJECT_SUFFIX
from tests.cpu.tiny_training import experiment
from tests.cpu.tiny_training.experiment import (
    MAX_STALENESS_STEPS,
    N_SAMPLES_PER_PROMPT,
    OBJECT_STORE_PAYLOADS,
    TRAIN_BATCH_SIZE,
    RolloutShape,
    TrainingMode,
    read_metrics,
    run_experiment,
    sampling_kind,
)
from tests.cpu.tiny_training.tiny_model import CURRICULUM_BINS, build_tiny_policy

pytestmark = pytest.mark.slow

NUM_STEPS = 3
RESUMED_STEP = 1
# A run takes under half a minute on an idle host. The margin absorbs slower CI hosts and concurrent pytest-xdist
# workers, and stays above the experiment's admission stall timeout so a stall reports its own error.
RUN_TIMEOUT_SECONDS = 300


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_policy(tmp_path_factory.mktemp("tiny_policy"))


@pytest.fixture(scope="module")
def runs() -> ForkServerContext:
    """Start each run in a fresh process, forked from a server that imported the training stack once."""
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload([experiment.__name__])
    return context


def _train(
    runs: ForkServerContext,
    root: Path,
    model: Path,
    mode: TrainingMode,
    shape: RolloutShape,
    *,
    steps: int,
    checkpoint_interval: int = -1,
) -> None:
    run = runs.Process(
        target=run_experiment,
        args=(root, model, mode, shape),
        kwargs={"max_steps": steps, "checkpoint_interval": checkpoint_interval},
    )
    run.start()
    run.join(RUN_TIMEOUT_SECONDS)
    if run.exitcode is None:
        run.kill()
        run.join()
        pytest.fail(f"the run did not finish within {RUN_TIMEOUT_SECONDS} seconds")
    assert run.exitcode == 0


def _trained_steps(root: Path) -> list[dict]:
    return [record for record in read_metrics(root) if "policy/raw_grad_norm" in record]


def _assert_trained_to_max_steps(root: Path, mode: TrainingMode, shape: RolloutShape) -> None:
    steps = _trained_steps(root)
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
    if OBJECT_STORE_PAYLOADS[mode]:
        # Every trained group was read back from its object, and the objects stay for groups generated but not
        # trained.
        objects = list((root / "rollouts").glob(f"*{ROLLOUT_OBJECT_SUFFIX}"))
        assert len(objects) >= NUM_STEPS * TRAIN_BATCH_SIZE


# The resume test covers asynchronous single-turn training.
@pytest.mark.parametrize(
    ("mode", "shape"),
    [
        (TrainingMode.SYNC, RolloutShape.SINGLE_TURN),
        (TrainingMode.SYNC, RolloutShape.STEP_WISE),
        (TrainingMode.ASYNC, RolloutShape.STEP_WISE),
    ],
)
def test_tiny_policy_trains_to_max_steps(
    runs: ForkServerContext, tmp_path: Path, tiny_policy: Path, mode: TrainingMode, shape: RolloutShape
):
    _train(runs, tmp_path, tiny_policy, mode, shape, steps=NUM_STEPS)

    _assert_trained_to_max_steps(tmp_path, mode, shape)


def test_async_training_resumes_with_committed_groups(runs: ForkServerContext, tmp_path: Path, tiny_policy: Path):
    mode, shape = TrainingMode.ASYNC, RolloutShape.SINGLE_TURN
    _train(runs, tmp_path, tiny_policy, mode, shape, steps=RESUMED_STEP, checkpoint_interval=1)
    # The first two batches' leases open together, so the second batch's groups commit while the first trains
    # and its checkpoint holds them. A later step's leases open only when its batch is taken, and whether they
    # commit before that step's checkpoint is a race.
    state = torch.load(tmp_path / "ckpts" / f"global_step_{RESUMED_STEP}" / "data.pt", weights_only=False)
    assert state.ready

    _train(runs, tmp_path, tiny_policy, mode, shape, steps=NUM_STEPS, checkpoint_interval=1)

    _assert_trained_to_max_steps(tmp_path, mode, shape)


def test_one_step_is_independent_of_micro_batch_size(runs: ForkServerContext, tmp_path: Path, tiny_policy: Path):
    roots = [tmp_path / "micro_8", tmp_path / "micro_4"]
    for root, micro_batch_size in zip(roots, [8, 4], strict=True):
        run = runs.Process(target=run_fixed_update, args=(root, tiny_policy, micro_batch_size))
        run.start()
        run.join(RUN_TIMEOUT_SECONDS)
        if run.exitcode is None:
            run.kill()
            run.join()
            pytest.fail(f"the update did not finish within {RUN_TIMEOUT_SECONDS} seconds")
        assert run.exitcode == 0
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
