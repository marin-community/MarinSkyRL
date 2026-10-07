"""End-to-end CPU DPO training on synthetic preference pairs through the production trainer."""

import math
import multiprocessing
from multiprocessing.context import ForkServerContext
from pathlib import Path

import pytest

from tests.cpu.tiny_training import preference_pairs
from tests.cpu.tiny_training.experiment import read_metrics
from tests.cpu.tiny_training.tiny_model import build_tiny_policy

pytestmark = pytest.mark.slow

NUM_STEPS = 6
RUN_TIMEOUT_SECONDS = 300


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_policy(tmp_path_factory.mktemp("tiny_dpo_policy"))


@pytest.fixture(scope="module")
def runs() -> ForkServerContext:
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload([preference_pairs.__name__])
    return context


def _run(context: ForkServerContext, root: Path, model: Path, steps: int) -> None:
    run = context.Process(
        target=preference_pairs.run_dpo_experiment, args=(root, model), kwargs={"steps": steps}
    )
    run.start()
    run.join(RUN_TIMEOUT_SECONDS)
    if run.exitcode is None:
        run.kill()
        run.join()
        pytest.fail(f"the DPO run did not finish within {RUN_TIMEOUT_SECONDS} seconds")
    assert run.exitcode == 0


def test_dpo_training_widens_the_preference_margin(runs, tmp_path, tiny_policy):
    _run(runs, tmp_path, tiny_policy, NUM_STEPS)

    records = [record for record in read_metrics(tmp_path) if "policy/dpo/loss" in record]
    assert [record["trainer/global_step"] for record in records] == list(range(1, NUM_STEPS + 1))
    losses = [record["policy/dpo/loss"] for record in records]
    margins = [record["policy/dpo/margin"] for record in records]
    accuracies = [record["policy/dpo/accuracy"] for record in records]
    assert all(math.isfinite(loss) for loss in losses)
    assert margins[-1] > margins[0], f"DPO did not widen the margin: {margins}"
    assert accuracies[-1] >= accuracies[0], f"DPO did not improve pair accuracy: {accuracies}"
    assert losses[-1] < losses[0], f"DPO loss did not decrease: {losses}"
    # The surrogate policy row carries the gradient, not the literal loss value.
    policy_rows = [record["policy/policy_loss"] for record in records]
    assert all(math.isfinite(row) for row in policy_rows)
