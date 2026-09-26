"""The coordinator loop between the group loader, rollout workers, and the rollout buffer actor."""

import asyncio
from collections import defaultdict

import pytest

from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import (
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutGroup,
    RolloutTask,
    RolloutWriter,
)
from skyrl_train.rollouts.context import RolloutRequestSpec, TrainingContext
from skyrl_train.rollouts.loader import GroupLoader

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

    def collate_fn(self, items: list[str]) -> list[dict]:
        return [{"uid": uid, "prompt": [], "env_class": None, "env_extras": {}} for uid in items]


class _Workers:
    """Generate groups in-process. Prompts in ``blocked`` never finish; prompts in ``failing`` raise."""

    def __init__(self, *, blocked: frozenset[str] = frozenset(), failing: frozenset[str] = frozenset()):
        self._blocked = blocked
        self._failing = failing
        self.started: list[str] = []
        self.written: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        uid = task.prompt["uid"]
        self.started.append(uid)
        if uid in self._blocked:
            await asyncio.Event().wait()
        if uid in self._failing:
            raise RuntimeError(f"rollout {uid} failed")
        batch = {
            "prompt_token_ids": [[1]] * SAMPLES_PER_PROMPT,
            "response_ids": [[2]] * SAMPLES_PER_PROMPT,
            "rewards": [0.0, 1.0],
            "loss_masks": [[1]] * SAMPLES_PER_PROMPT,
            "rollout_logprobs": None,
        }
        await writer.write_rollout(
            task.lease, RolloutGroup(batch, uid, task.lease.policy_step, task.prompt, task.request)
        )
        self.written[uid].set()
        return SAMPLES_PER_PROMPT


def _context(uids: list[str], workers: _Workers, *, batch_size: int, max_in_flight: int) -> TrainingContext:
    return TrainingContext(
        GroupLoader(_Prompts(uids), seed=0, shuffle=False),
        RolloutBufferConfig(batch_size, max_in_flight, 1, None, None),
        CONTENT_POLICY,
        RolloutRequestSpec(samples_per_prompt=SAMPLES_PER_PROMPT, sampling_params={}, environment_class="test"),
        workers,
        rollout_spans=False,
    )


async def _ignore(groups: list[RolloutGroup]) -> None:
    pass


async def _next_uids(context: TrainingContext) -> list[str]:
    groups, _ = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
    return [group.uid for group in groups]


@pytest.mark.asyncio
async def test_slow_rollout_does_not_block_training_at_positive_staleness(ray_module):
    workers = _Workers(blocked=frozenset({"slow"}))
    context = _context(["slow", "a", "b"], workers, batch_size=1, max_in_flight=2)
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
async def test_resume_regenerates_uncommitted_prompts_and_keeps_committed_groups(ray_module):
    workers = _Workers(blocked=frozenset({"b"}))
    context = _context(["a", "b", "c"], workers, batch_size=1, max_in_flight=2)
    context.start()
    try:
        await context.publish(1)
        assert await _next_uids(context) == ["a"]
        await context.publish(2)
        await asyncio.wait_for(workers.written["c"].wait(), STALL_TIMEOUT)
        state = await context.state_dict()
    finally:
        await context.close()

    assert [rollout.payload[0].uid for rollout in state.ready] == ["c"]
    assert [prompt["uid"] for prompt in state.loader.retries] == ["b"]

    resumed_workers = _Workers()
    resumed = _context(["a", "b", "c"], resumed_workers, batch_size=1, max_in_flight=2)
    await resumed.load_state_dict(state)
    resumed.start()
    try:
        await resumed.publish(2)
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
