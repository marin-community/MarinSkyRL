"""The coordinator loop between the group loader, rollout workers, and the rollout buffer actor."""

from copy import deepcopy
import asyncio
import inspect
from collections import defaultdict
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import ray
from ray.actor import ActorHandle

from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy, resolve_dynamic_sampling_criteria
from skyrl_train.group_admission import (
    AdmissionRejection,
    GroupAdmissionPolicy,
    GroupAdvantageInvariant,
    TrainingGroupInvariantError,
)
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    ReadyRollout,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutGroup,
    RolloutTask,
    RolloutWriter,
)
from skyrl_train.rollouts.context import (
    RolloutRequestSpec,
    TrainingContext,
    TrainingContextState,
    start_rollout_buffer,
)
from skyrl_train.rollouts.loader import PromptLoader, PromptLoaderState, JudgedGroup, PromptOrder, SeededPasses
from skyrl_train.rollouts.payloads import MemoryPayloads, ObjectStorePayloads, PayloadStore
from skyrl_train.utils import validate_cfg
from tests.cpu.test_teacher_config_rejection import selected_topk_config

SAMPLES_PER_PROMPT = 2
STALL_TIMEOUT = 10.0
CONTENT_POLICY = RolloutContentPolicy(
    GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=SAMPLES_PER_PROMPT),
        rollout_logprobs_required=False,
    ),
    GroupSelectionPolicy(None),
)


class _Prompts:
    def __init__(self, uids: list[str]):
        self._uids = uids

    def __len__(self) -> int:
        return len(self._uids)

    def __getitem__(self, index: int) -> str:
        return self._uids[index]

    def uid(self, index: int) -> str:
        return self._uids[index]

    def collate_fn(self, items: list[str]) -> list[dict]:
        return [_prompt(uid) for uid in items]


def _prompt(uid: str) -> dict:
    return {"uid": uid, "prompt": [], "env_class": None, "env_extras": {}}


def _batch(*, masked: bool = False) -> dict:
    return {
        "prompt_token_ids": [[1]] * SAMPLES_PER_PROMPT,
        "response_ids": [[2]] * SAMPLES_PER_PROMPT,
        "rewards": [0.0, 1.0],
        "loss_masks": [[0 if masked else 1]] * SAMPLES_PER_PROMPT,
        "rollout_logprobs": None,
    }


class _Workers:
    """Generate groups in-process.

    Prompts in ``blocked`` wait for ``unblocked``, prompts in ``failing`` raise, and the first group of each prompt
    in ``masked_once`` has no trainable tokens.
    """

    def __init__(
        self,
        *,
        blocked: frozenset[str] = frozenset(),
        failing: frozenset[str] = frozenset(),
        masked_once: frozenset[str] = frozenset(),
    ):
        self._blocked = blocked
        self._failing = failing
        self._masked_once = set(masked_once)
        self.started: list[str] = []
        # Rows started while any blocked row was still waiting.
        self.started_while_blocked: list[str] = []
        self.unblocked: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.written: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        uid = task.prompt["uid"]
        self.started.append(uid)
        if not all(self.unblocked[blocked].is_set() for blocked in self._blocked):
            self.started_while_blocked.append(uid)
        if uid in self._blocked:
            await self.unblocked[uid].wait()
        if uid in self._failing:
            raise RuntimeError(f"rollout {uid} failed")
        masked = uid in self._masked_once
        self._masked_once.discard(uid)
        await writer.write_rollout(
            task.lease, RolloutGroup(_batch(masked=masked), uid, task.lease.policy_step, task.prompt)
        )
        self.written[uid].set()
        return SAMPLES_PER_PROMPT


class _RecordingOrder:
    """Dataset order that records the judged groups of each step."""

    def __init__(self, num_rows: int):
        self._passes = SeededPasses(num_rows, seed=0, shuffle=False)
        self.observed: list[list[JudgedGroup]] = []

    def next_index(self) -> int:
        return self._passes.next_index()

    def observe(self, groups):
        self.observed.append(list(groups))
        return {"order/groups": float(len(groups))}

    def state_dict(self):
        return self._passes.state_dict()

    def load_state_dict(self, state) -> None:
        self._passes.load_state_dict(state)


class _InProcessActor:
    """Serves an actor's ``method.remote(...)`` calls on the caller's event loop.

    Starting a real buffer actor spawns a Ray worker that imports the trainer, which takes seconds per test.
    """

    def __init__(self, instance):
        self._instance = instance

    def __getattr__(self, name: str) -> SimpleNamespace:
        method = getattr(self._instance, name)

        async def call(*args):
            result = method(*args)
            return await result if inspect.isawaitable(result) else result

        return SimpleNamespace(remote=lambda *args: asyncio.ensure_future(call(*args)))


def _in_process_buffer(config: RolloutBufferConfig) -> _InProcessActor:
    return _InProcessActor(RolloutBuffer(config))


StartBuffer = Callable[[RolloutBufferConfig], ActorHandle]


@pytest.fixture
def start_buffer(request, monkeypatch) -> StartBuffer:
    """Start each context's buffer in-process, or as a Ray actor when parametrized with ``"ray_actor"``."""
    if getattr(request, "param", "in_process") == "ray_actor":
        return start_rollout_buffer
    # An in-process buffer has no actor for ``TrainingContext.close`` to kill.
    monkeypatch.setattr(ray, "kill", lambda _actor: None)
    return _in_process_buffer


def _context(
    uids: list[str],
    workers: _Workers,
    start_buffer: StartBuffer,
    *,
    batch_size: int,
    max_in_flight: int,
    max_staleness_steps: int = 1,
    batch_policy: BatchPolicy = BatchPolicy.FULL_BATCH,
    order: PromptOrder | None = None,
    payloads: PayloadStore | None = None,
) -> TrainingContext:
    config = RolloutBufferConfig(batch_size, max_in_flight, max_staleness_steps, batch_policy, None, None)
    return TrainingContext(
        PromptLoader(_Prompts(uids), order or SeededPasses(len(uids), seed=0, shuffle=False), batch_size=batch_size),
        config,
        start_buffer(config),
        CONTENT_POLICY,
        RolloutRequestSpec(samples_per_prompt=SAMPLES_PER_PROMPT, sampling_params={}, environment_class="test"),
        workers,
        payloads or MemoryPayloads(),
        rollout_spans=False,
    )


async def _ignore(groups: list[RolloutGroup]) -> None:
    pass


async def _next_uids(context: TrainingContext) -> list[str]:
    groups, _ = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
    return [group.uid for group in groups]


@pytest.mark.asyncio
async def test_rolling_batches_do_not_wait_for_a_slow_rollout(ray_module, start_buffer):
    workers = _Workers(blocked=frozenset({"slow"}))
    context = _context(
        ["slow", "a", "b"], workers, start_buffer, batch_size=1, max_in_flight=2, batch_policy=BatchPolicy.ROLLING
    )
    context.start()
    try:
        await context.publish(1)
        assert await _next_uids(context) == ["a"]
        await context.publish(2)
        assert await _next_uids(context) == ["b"]
        assert "slow" in workers.started
    finally:
        await context.close()


@pytest.mark.asyncio
async def test_full_batches_train_a_slow_rollout_in_its_own_step(ray_module, start_buffer):
    workers = _Workers(blocked=frozenset({"slow"}))
    context = _context(["slow", "a", "b"], workers, start_buffer, batch_size=1, max_in_flight=2)
    context.start()
    try:
        await context.publish(1)
        await asyncio.wait_for(workers.written["a"].wait(), STALL_TIMEOUT)
        workers.unblocked["slow"].set()
        assert await _next_uids(context) == ["slow"]
        await context.publish(2)
        assert await _next_uids(context) == ["a"]
    finally:
        await context.close()


@pytest.mark.asyncio
async def test_a_slow_row_does_not_regenerate_rows_already_in_the_batch(ray_module, start_buffer):
    workers = _Workers(blocked=frozenset({"b"}))
    context = _context(["a", "b", "c"], workers, start_buffer, batch_size=3, max_in_flight=6)
    context.start()
    try:
        await context.publish(1)
        await asyncio.wait_for(workers.written["a"].wait(), STALL_TIMEOUT)
        await asyncio.wait_for(workers.written["c"].wait(), STALL_TIMEOUT)
        workers.unblocked["b"].set()
        groups, metrics = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
    finally:
        await context.close()

    assert sorted(group.uid for group in groups) == ["a", "b", "c"]
    assert metrics["async/rejected_count/duplicate_uid"] == 0
    assert workers.started_while_blocked == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_a_rejected_row_is_generated_again_within_a_synchronous_step(ray_module, start_buffer):
    workers = _Workers(masked_once=frozenset({"m"}))
    context = _context(["m", "a"], workers, start_buffer, batch_size=2, max_in_flight=2, max_staleness_steps=0)
    context.start()
    try:
        await context.publish(1)
        groups, metrics = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
    finally:
        await context.close()

    assert sorted(group.uid for group in groups) == ["a", "m"]
    assert metrics["async/rejected_count/fully_masked"] == 1
    assert workers.started == ["m", "a", "m"]


@pytest.mark.asyncio
async def test_resume_does_not_regenerate_a_committed_group(ray_module, start_buffer):
    group = RolloutGroup(_batch(), "c", 1, _prompt("c"))
    committed = ReadyRollout("committed", 1, 1, group.prompt, CONTENT_POLICY.verdict(group), [group], None)
    # The order's next draw is row "c", which the checkpoint already holds.
    state = TrainingContextState(PromptLoaderState({"epoch": 0, "position": 2}, []), [committed], None)
    workers = _Workers()
    context = _context(["a", "b", "c"], workers, start_buffer, batch_size=2, max_in_flight=4)
    await context.load_state_dict(state)
    context.start()
    try:
        await context.publish(1)
        groups, metrics = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
    finally:
        await context.close()

    assert groups[0].uid == "c"
    assert metrics["async/rejected_count/duplicate_uid"] == 0
    assert workers.started[:2] == ["a", "b"]


@pytest.mark.asyncio
async def test_each_batch_reports_its_judged_groups_to_the_prompt_order(ray_module, start_buffer):
    order = _RecordingOrder(3)
    context = _context(["a", "b", "c"], _Workers(), start_buffer, batch_size=2, max_in_flight=2, order=order)
    context.start()
    try:
        await context.publish(1)
        groups, metrics = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
    finally:
        await context.close()

    assert order.observed == [[JudgedGroup(group.uid, (0.0, 1.0)) for group in groups]]
    assert metrics["order/groups"] == 2.0


async def _checkpoint_with_committed_group(payloads: PayloadStore, start_buffer: StartBuffer) -> TrainingContextState:
    """Train "a", then checkpoint once "c" is committed while "b" is still generating."""
    workers = _Workers(blocked=frozenset({"b"}))
    context = _context(["a", "b", "c"], workers, start_buffer, batch_size=1, max_in_flight=2, payloads=payloads)
    context.start()
    try:
        await context.publish(1)
        assert await _next_uids(context) == ["a"]
        await context.publish(2)
        await asyncio.wait_for(workers.written["c"].wait(), STALL_TIMEOUT)
        return await context.state_dict()
    finally:
        await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload_kind", "start_buffer"),
    # The real actor covers Ray's handling of in-memory payload references across the actor boundary.
    [("memory", "ray_actor"), ("object_store", "in_process")],
    indirect=["start_buffer"],
)
async def test_resume_regenerates_uncommitted_prompts_and_keeps_committed_groups(
    ray_module, payload_kind, start_buffer, tmp_path
):
    payloads = MemoryPayloads() if payload_kind == "memory" else ObjectStorePayloads(str(tmp_path / "rollouts"))
    state = await _checkpoint_with_committed_group(payloads, start_buffer)

    assert [rollout.verdict.uid for rollout in state.ready] == ["c"]
    assert [prompt["uid"] for prompt in state.loader.retries] == ["b"]

    resumed_workers = _Workers()
    resumed = _context(["a", "b", "c"], resumed_workers, start_buffer, batch_size=1, max_in_flight=2, payloads=payloads)
    await resumed.load_state_dict(state)
    resumed.start()
    try:
        # "b" was leased for step 2 and "c" for step 3, so each trains in its own step.
        await resumed.publish(2)
        assert await _next_uids(resumed) == ["b"]
        await resumed.publish(3)
        assert await _next_uids(resumed) == ["c"]
        assert resumed_workers.started[0] == "b"
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_failed_rollout_fails_the_next_batch(ray_module, start_buffer):
    context = _context(["bad"], _Workers(failing=frozenset({"bad"})), start_buffer, batch_size=1, max_in_flight=1)
    context.start()
    try:
        await context.publish(1)
        with pytest.raises(RuntimeError, match="rollout bad failed"):
            await _next_uids(context)
    finally:
        await context.close()


@pytest.mark.asyncio
async def test_resume_rejects_committed_groups_from_another_payload_store(ray_module, tmp_path, start_buffer):
    state = await _checkpoint_with_committed_group(ObjectStorePayloads(str(tmp_path / "rollouts")), start_buffer)

    resumed = _context(["a", "b", "c"], _Workers(), start_buffer, batch_size=1, max_in_flight=2)
    try:
        with pytest.raises(ValueError, match="object_store_root"):
            await resumed.load_state_dict(state)
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_resume_under_a_new_object_store_root_trains_the_checkpointed_objects(ray_module, tmp_path, start_buffer):
    state = await _checkpoint_with_committed_group(ObjectStorePayloads(str(tmp_path / "attempt-1")), start_buffer)

    payloads = ObjectStorePayloads(str(tmp_path / "attempt-2"))
    resumed = _context(["a", "b", "c"], _Workers(), start_buffer, batch_size=1, max_in_flight=2, payloads=payloads)
    await resumed.load_state_dict(state)
    resumed.start()
    try:
        await resumed.publish(2)
        await _next_uids(resumed)
        await resumed.publish(3)
        assert await _next_uids(resumed) == ["c"]
    finally:
        await resumed.close()


@pytest.fixture(params=["memory", "object_store"])
def payloads(request, tmp_path) -> PayloadStore:
    if request.param == "memory":
        return MemoryPayloads()
    return ObjectStorePayloads(str(tmp_path / "rollouts"))


@pytest.mark.asyncio
@pytest.mark.parametrize(("rejected", "accepted"), [([0.5], [0.25]), ([0.8, 1.0], [0.0, 0.5])])
async def test_writer_filters_success_ceiling_using_final_outcomes(ray_module, payloads, rejected, accepted):
    policy = RolloutContentPolicy(
        GroupAdmissionPolicy(
            GroupAdvantageInvariant.no_group_advantage(physical_group_size=2),
            rollout_logprobs_required=False,
        ),
        GroupSelectionPolicy(
            DynamicSamplingType.FILTER,
            criteria=resolve_dynamic_sampling_criteria(max_mean_reward=0.5),
        ),
    )
    buffer = ray.remote(RolloutBuffer).remote(
        RolloutBufferConfig(1, 3, 1, BatchPolicy.FULL_BATCH, DynamicSamplingType.FILTER, 3)
    )
    writer = payloads.writer(buffer, policy)
    try:
        await buffer.publish.remote(1)
        for uid, rewards, final in [
            ("easy", [0.0, *rejected], [False, *([True] * len(rejected))]),
            ("keep", [100.0, *accepted], [False, *([True] * len(accepted))]),
        ]:
            rows = len(rewards)
            batch = dict(
                prompt_token_ids=[[1]] * rows,
                response_ids=[[2]] * rows,
                loss_masks=[[1]] * rows,
                rewards=rewards,
                is_last_step=final,
            )
            lease = await buffer.acquire_lease.remote()
            await writer.write_rollout(lease, RolloutGroup(batch, uid, 1, _prompt(uid)))
        admitted = []
        while True:
            admission = await buffer.admit.remote(STALL_TIMEOUT)
            admitted.extend(await payloads.fetch(admission.payloads))
            if admission.selection is not None:
                break
        assert [group.uid for group in admitted] == ["keep"]
        assert admission.selection.metrics["async/dynamic_sampling/discarded_count"] == 1
    finally:
        ray.kill(buffer)


class _EvidenceWorkers:
    def __init__(self, batch: dict, missing_field: str):
        self.batch = batch
        self.missing_field = missing_field
        self.rejections = ()

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        invalid = deepcopy(self.batch)
        invalid[self.missing_field][0][0] = -1 if self.missing_field == "student_topk_indices" else np.nan
        try:
            await writer.write_rollout(
                task.lease, RolloutGroup(invalid, task.prompt["uid"], task.lease.policy_step, task.prompt)
            )
        except TrainingGroupInvariantError as error:
            self.rejections = error.rejections
        else:
            return SAMPLES_PER_PROMPT
        await writer.write_rollout(
            task.lease, RolloutGroup(self.batch, task.prompt["uid"], task.lease.policy_step, task.prompt)
        )
        return SAMPLES_PER_PROMPT


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_field", ["rollout_logprobs", "student_topk_indices", "behavior_topk_logprobs"])
async def test_configured_context_requires_behavior_evidence_only_at_trainable_tokens(
    ray_module, payloads, missing_field
):
    cfg = selected_topk_config()
    cfg.teachers.primary.top_k = 2
    cfg.generator.sampling_params.logprobs = 2
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.n_samples_per_prompt = SAMPLES_PER_PROMPT
    cfg.trainer.train_batch_size = 1
    cfg.trainer.policy_mini_batch_size = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.placement.ref_num_gpus_per_node = 1
    cfg.trainer.rollout_buffer.max_in_flight = 1
    cfg.trainer.rollout_buffer.object_store_root = payloads.object_store_root
    if missing_field == "rollout_logprobs":
        cfg.trainer.algorithm.distillation.reward_mode = "add"
        cfg.trainer.algorithm.off_policy_correction = "tis"
    validate_cfg(cfg)
    batch = _batch()
    batch.update(
        response_ids=[[2, 3], [4, 5]],
        loss_masks=[[1, 0], [0, 0]],
        rollout_logprobs=[np.asarray([-0.2, np.nan], dtype=np.float32), np.full(2, np.nan, dtype=np.float32)],
        student_topk_indices=[np.asarray([[1, 2], [-1, -1]], dtype=np.int32), np.full((2, 2), -1, dtype=np.int32)],
        behavior_topk_logprobs=[
            np.asarray([[-0.3, -1.4], [np.nan, np.nan]], dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
        ],
    )
    if missing_field != "rollout_logprobs":
        batch["rollout_logprobs"] = None
    workers = _EvidenceWorkers(batch, missing_field)
    context = TrainingContext.from_config(cfg, _Prompts(["a"]), workers)
    context.start()
    try:
        await context.publish(1)
        groups, _ = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
        expected = (
            AdmissionRejection.MISSING_ROLLOUT_LOGPROBS
            if missing_field == "rollout_logprobs"
            else AdmissionRejection.MISSING_BEHAVIOR_TOPK
        )
        assert workers.rejections == (expected,)
        np.testing.assert_equal([group.trajectory_batch for group in groups], [batch])
    finally:
        await context.close()
