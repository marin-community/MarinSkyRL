"""The coordinator loop between the group loader, rollout workers, and the rollout buffer actor."""

import asyncio
from collections import defaultdict

import pytest

from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    ReadyRollout,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutGroup,
    RolloutTask,
    RolloutWriter,
)
from skyrl_train.rollouts.context import RolloutRequestSpec, RolloutResumePolicy, TrainingContext, TrainingContextState
from skyrl_train.rollouts.loader import PromptLoader, PromptLoaderState, JudgedGroup, PromptOrder, SeededPasses
from skyrl_train.rollouts.payloads import MemoryPayloads, ObjectStorePayloads, PayloadStore

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


def _context(
    uids: list[str],
    workers: _Workers,
    *,
    batch_size: int,
    max_in_flight: int,
    max_staleness_steps: int = 1,
    batch_policy: BatchPolicy = BatchPolicy.FULL_BATCH,
    order: PromptOrder | None = None,
    payloads: PayloadStore | None = None,
) -> TrainingContext:
    return TrainingContext(
        PromptLoader(_Prompts(uids), order or SeededPasses(len(uids), seed=0, shuffle=False), batch_size=batch_size),
        RolloutBufferConfig(batch_size, max_in_flight, max_staleness_steps, batch_policy, None, None),
        CONTENT_POLICY,
        RolloutRequestSpec(samples_per_prompt=SAMPLES_PER_PROMPT, sampling_params={}, environment_class="test"),
        workers,
        payloads or MemoryPayloads(),
        rollout_spans=False,
    )


@pytest.fixture(params=["memory", "object_store"])
def payloads(request, tmp_path) -> PayloadStore:
    if request.param == "memory":
        return MemoryPayloads()
    return ObjectStorePayloads(str(tmp_path / "rollouts"))


async def _ignore(groups: list[RolloutGroup]) -> None:
    pass


async def _next_uids(context: TrainingContext) -> list[str]:
    groups, _ = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
    return [group.uid for group in groups]


@pytest.mark.asyncio
async def test_rolling_batches_do_not_wait_for_a_slow_rollout(ray_module):
    workers = _Workers(blocked=frozenset({"slow"}))
    context = _context(["slow", "a", "b"], workers, batch_size=1, max_in_flight=2, batch_policy=BatchPolicy.ROLLING)
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
async def test_full_batches_train_a_slow_rollout_in_its_own_step(ray_module):
    workers = _Workers(blocked=frozenset({"slow"}))
    context = _context(["slow", "a", "b"], workers, batch_size=1, max_in_flight=2)
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
async def test_a_slow_row_does_not_regenerate_rows_already_in_the_batch(ray_module):
    workers = _Workers(blocked=frozenset({"b"}))
    context = _context(["a", "b", "c"], workers, batch_size=3, max_in_flight=6)
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
async def test_a_rejected_row_is_generated_again_within_a_synchronous_step(ray_module):
    workers = _Workers(masked_once=frozenset({"m"}))
    context = _context(["m", "a"], workers, batch_size=2, max_in_flight=2, max_staleness_steps=0)
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
async def test_resume_does_not_regenerate_a_committed_group(ray_module):
    group = RolloutGroup(_batch(), "c", 1, _prompt("c"))
    committed = ReadyRollout("committed", 1, 1, group.prompt, CONTENT_POLICY.verdict(group), [group], None)
    # The order's next draw is row "c", which the checkpoint already holds.
    state = TrainingContextState(PromptLoaderState({"epoch": 0, "position": 2}, []), [committed], None)
    workers = _Workers()
    context = _context(["a", "b", "c"], workers, batch_size=2, max_in_flight=4)
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
async def test_each_batch_reports_its_judged_groups_to_the_prompt_order(ray_module):
    order = _RecordingOrder(3)
    context = _context(["a", "b", "c"], _Workers(), batch_size=2, max_in_flight=2, order=order)
    context.start()
    try:
        await context.publish(1)
        groups, metrics = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
    finally:
        await context.close()

    assert order.observed == [[JudgedGroup(group.uid, (0.0, 1.0)) for group in groups]]
    assert metrics["order/groups"] == 2.0


async def _checkpoint_with_committed_group(payloads: PayloadStore) -> TrainingContextState:
    """Train "a", then checkpoint once "c" is committed while "b" is still generating."""
    workers = _Workers(blocked=frozenset({"b"}))
    context = _context(["a", "b", "c"], workers, batch_size=1, max_in_flight=2, payloads=payloads)
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
async def test_resume_regenerates_uncommitted_prompts_and_keeps_committed_groups(ray_module, payloads):
    state = await _checkpoint_with_committed_group(payloads)

    assert [rollout.verdict.uid for rollout in state.ready] == ["c"]
    assert [prompt["uid"] for prompt in state.loader.retries] == ["b"]

    resumed_workers = _Workers()
    resumed = _context(["a", "b", "c"], resumed_workers, batch_size=1, max_in_flight=2, payloads=payloads)
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
async def test_failed_rollout_fails_the_next_batch(ray_module):
    context = _context(["bad"], _Workers(failing=frozenset({"bad"})), batch_size=1, max_in_flight=1)
    context.start()
    try:
        await context.publish(1)
        with pytest.raises(RuntimeError, match="rollout bad failed"):
            await _next_uids(context)
    finally:
        await context.close()


@pytest.mark.asyncio
async def test_resume_rejects_committed_groups_from_another_payload_store(ray_module, tmp_path):
    state = await _checkpoint_with_committed_group(ObjectStorePayloads(str(tmp_path / "rollouts")))

    resumed = _context(["a", "b", "c"], _Workers(), batch_size=1, max_in_flight=2)
    try:
        with pytest.raises(ValueError, match="object_store_root"):
            await resumed.load_state_dict(state)
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_resume_under_a_new_object_store_root_trains_the_checkpointed_objects(ray_module, tmp_path):
    state = await _checkpoint_with_committed_group(ObjectStorePayloads(str(tmp_path / "attempt-1")))

    payloads = ObjectStorePayloads(str(tmp_path / "attempt-2"))
    resumed = _context(["a", "b", "c"], _Workers(), batch_size=1, max_in_flight=2, payloads=payloads)
    await resumed.load_state_dict(state)
    resumed.start()
    try:
        await resumed.publish(2)
        await _next_uids(resumed)
        await resumed.publish(3)
        assert await _next_uids(resumed) == ["c"]
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_resume_regenerates_old_verdicts_without_restarting_prompt_order(ray_module):
    group = RolloutGroup(_batch(masked=True), "c", 1, _prompt("c"))
    committed = ReadyRollout("old-verdict", 1, 1, group.prompt, CONTENT_POLICY.verdict(group), [], None)
    state = TrainingContextState(PromptLoaderState({"epoch": 0, "position": 3}, [_prompt("b")]), [committed], None)
    workers = _Workers()
    context = _context(["a", "b", "c", "d"], workers, batch_size=2, max_in_flight=2)
    await context.load_state_dict(state, committed_groups=RolloutResumePolicy.REGENERATE)
    assert context.loader.state_dict().order == state.loader.order
    context.start()
    try:
        await context.publish(4)
        groups, metrics = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
    finally:
        await context.close()
    assert workers.started[:2] == ["b", "c"]
    assert sorted(g.uid for g in groups) == ["b", "c"]
    assert metrics.get("async/rejected_count/fully_masked", 0) == 0
    assert state.ready == [committed]
