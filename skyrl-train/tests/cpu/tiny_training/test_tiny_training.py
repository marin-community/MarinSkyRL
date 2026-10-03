"""End-to-end CPU training through the production entrypoints on a tiny policy."""

import multiprocessing
import json
import random
from multiprocessing.context import ForkServerContext
from pathlib import Path

import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from omegaconf import OmegaConf
from skyrl_train.callbacks.base import TrainerCallback
from skyrl_train.callbacks.builtin import register_callback
from skyrl_train.callbacks.types import CallbackErrorBehavior

from marinskyrl.checkpoint_paths import POLICY_CHECKPOINT_SUBDIRECTORY
from skyrl_train.checkpoint_generation import resolve_checkpoint_payload
from skyrl_train.hf_export_schema import TRAINER_STATE_FILENAME
from tests.checkpoint_parity_state import assert_state_equal, snapshot_value
from tests.cpu.tiny_training.cpu_backend import CHECKPOINT_FILE_TEMPLATE, CausalLMPolicy
from tests.cpu.tiny_training.fixed_batch import fixed_training_batch, run_fixed_update
from skyrl_train.rollouts.payloads import ROLLOUT_OBJECT_SUFFIX
from skyrl_train.training_batch import TrainingInputBatch
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


@register_callback("test_step_limit")
class _StepLimit(TrainerCallback):
    error_behavior = CallbackErrorBehavior.RAISE

    def __init__(self, limit: int, action: str = "continue"):
        self.limit = limit
        self.action = action
        self.failed = False

    def on_train_begin(self, state, control, **kwargs):
        control.step_limit = self.limit
        trainer = kwargs["trainer"]
        path = Path(trainer.cfg.trainer.export_path) / f"start-{state.global_step}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checkpoint_path": trainer.loaded_checkpoint_path}))
        if self.action == "callback_stop" and trainer.loaded_checkpoint_path is not None:
            random.random()
            np.random.random()
            torch.rand(1)
            control.should_training_stop = True
        return control

    def on_step_end(self, state, control, **kwargs):
        if self.action == "ftpo_stop":
            kwargs["trainer"]._ftpo_stopped = True
            control.should_training_stop = True
        return control

    def on_save(self, state, control, **kwargs):
        if self.action == "save_stop":
            control.should_training_stop = True
        if self.action == "save_failure" and not self.failed:
            self.failed = True
            raise OSError("injected checkpoint callback storage failure")
        return control


def _run_limited_training(root: Path, model: Path, limits: tuple[int, ...], action: str) -> None:
    cfg = experiment.tiny_training_config(
        root, model, TrainingMode.SYNC, RolloutShape.SINGLE_TURN, max_steps=1, checkpoint_interval=1
    )
    OmegaConf.update(
        cfg,
        "trainer.callbacks",
        [{"type": "test_step_limit", "limit": limit, "action": action} for limit in limits]
        + [{"type": "checkpoint", "save_steps": 1}],
        force_add=True,
    )
    experiment.run_tiny_training(cfg)


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
        kwargs={
            "max_steps": steps,
            "checkpoint_interval": checkpoint_interval,
            "dump_data_batch": mode is TrainingMode.SYNC,
        },
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
    if mode is TrainingMode.SYNC:
        for step in range(1, NUM_STEPS + 1):
            batch = TrainingInputBatch().load(root / f"exports/dumped_data/global_step_{step}_training_input.pkl")
            eligible = batch["loss_mask"].bool()
            expected = torch.zeros_like(batch["action_log_probs"], dtype=torch.float32)
            expected[eligible] = (
                (batch["action_log_probs"][eligible].float() - batch["rollout_logprobs"][eligible].float())
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
    step_path = tmp_path / "ckpts" / f"global_step_{RESUMED_STEP}"
    state = torch.load(Path(resolve_checkpoint_payload(str(step_path))) / "data.pt", weights_only=False)
    assert state.ready

    _train(runs, tmp_path, tiny_policy, mode, shape, steps=NUM_STEPS, checkpoint_interval=1)

    _assert_trained_to_max_steps(tmp_path, mode, shape)


@pytest.mark.parametrize(
    ("action", "phases", "expected_steps"),
    [
        ("continue", ((1,), (3, 4), (3,)), [1, 2, 3]),
        ("ftpo_stop", ((1,), (3, 4)), [1]),
        ("callback_stop", ((1,), (3, 4)), [1]),
        ("save_stop", ((3,),), [1]),
        ("save_failure", ((1,),), [1]),
    ],
)
def test_checkpointed_training_obeys_callback_controls(runs, tmp_path, tiny_policy, action, phases, expected_steps):
    rng_states = []
    for limits in phases:
        run = runs.Process(target=_run_limited_training, args=(tmp_path, tiny_policy, limits, action))
        run.start()
        run.join(RUN_TIMEOUT_SECONDS)
        if run.exitcode is None:
            run.kill()
            run.join()
            pytest.fail("callback-limited training did not finish")
        assert run.exitcode == 0
        latest = int((tmp_path / "ckpts" / "latest_ckpt_global_step.txt").read_text())
        payload = Path(resolve_checkpoint_payload(str(tmp_path / "ckpts" / f"global_step_{latest}"), verify_files=True))
        rng_states.append(snapshot_value(torch.load(payload / "driver_rng_state.pt", weights_only=False)))
    assert [record["trainer/global_step"] for record in _trained_steps(tmp_path)] == expected_steps
    if action in ("ftpo_stop", "callback_stop"):
        assert_state_equal(rng_states[0], rng_states[1])
    if action == "continue":
        for step in (1, 3):
            provenance = json.loads((tmp_path / "exports" / f"start-{step}.json").read_text())
            state = torch.load(Path(provenance["checkpoint_path"]) / TRAINER_STATE_FILENAME, weights_only=False)
            assert state["global_step"] == step
    checkpoint_metrics = [row for row in read_metrics(tmp_path) if "timing/save_checkpoints" in row]
    assert checkpoint_metrics
    for row in checkpoint_metrics:
        assert row["timing/step_wall/checkpoint_work"] >= row["timing/save_checkpoints"] > 0
        assert row["timing/step"] >= row["timing/step_wall/checkpoint_work"]
    if action == "save_failure":
        assert any(row.get("trainer/checkpoint_save_failures") == 1 for row in checkpoint_metrics)


def test_one_step_is_independent_of_micro_batch_size(runs: ForkServerContext, tmp_path: Path, tiny_policy: Path):
    batch = fixed_training_batch(str(tiny_policy))
    model = CausalLMPolicy(AutoModelForCausalLM.from_pretrained(tiny_policy, dtype=torch.float32))
    log_probs = model(batch["sequences"], num_actions=4, attention_mask=batch["attention_mask"])
    ratios = (log_probs - batch["action_log_probs"]).exp()
    loss = (
        -(ratios * batch["advantages"] * batch["correction_weights"] * batch["loss_mask"]).sum()
        / batch["loss_mask"].sum()
    )
    loss.backward()
    reference_norm = torch.linalg.vector_norm(
        torch.stack([parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None])
    ).item()
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
    norms = [torch.load(root / "train_status.pt", weights_only=False)["raw_grad_norm"] for root in roots]
    for norm in norms:
        torch.testing.assert_close(norm, reference_norm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(norms[0], norms[1], rtol=1e-5, atol=1e-6)
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
