"""Lease accounting and batch selection of the rollout buffer, exercised in-process without Ray."""

import asyncio

import pytest

from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionResult
from skyrl_train.group_admission import AdmissionRejection, GroupAdmissionStalledError
from skyrl_train.rollouts.buffer import GroupRewards, RolloutBuffer, RolloutBufferConfig, RolloutVerdict
from skyrl_train.telemetry import GeneratedWork

# Long enough that an expected admission never times out, short enough that a blocked lease fails fast.
PROGRESS_TIMEOUT = 5.0
BLOCKED_TIMEOUT = 0.05
UNIFORM_REWARDS = GroupRewards(sample_count=2, optimization_reward_sum=0.0, outcome_reward_sum=0.0, passed=False)
SPREAD_REWARDS = GroupRewards(sample_count=2, optimization_reward_sum=1.0, outcome_reward_sum=1.0, passed=True)


def _buffer(
    *,
    batch_size: int,
    max_in_flight: int = 4,
    max_staleness_steps: int = 1,
    dynamic_sampling: DynamicSamplingType | None = None,
    max_candidate_groups: int | None = None,
) -> RolloutBuffer:
    return RolloutBuffer(
        RolloutBufferConfig(batch_size, max_in_flight, max_staleness_steps, dynamic_sampling, max_candidate_groups)
    )


def _verdict(
    uid: str,
    *,
    rejection: AdmissionRejection | None = None,
    selection: GroupSelectionResult = GroupSelectionResult.KEEP,
    rewards: GroupRewards | None = None,
) -> RolloutVerdict:
    rejections = (rejection,) if rejection is not None else ()
    return RolloutVerdict(uid, rejections, selection, rewards, GeneratedWork(2, 2, 8))


async def _commit(buffer: RolloutBuffer, lease_id: str, uid: str, **verdict) -> None:
    """Commit a group whose payload stands in for its object reference with its UID."""
    await buffer.commit(lease_id, {"uid": uid}, _verdict(uid, **verdict), [uid])


async def _generate(buffer: RolloutBuffer, uid: str, **verdict) -> None:
    lease = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    await _commit(buffer, lease.lease_id, uid, **verdict)


async def _take_batch(buffer: RolloutBuffer) -> tuple[list[str], dict[str, float]]:
    """Admit until the batch completes, returning its payloads in admission order and its metrics."""
    payloads = []
    while True:
        admission = await buffer.admit(PROGRESS_TIMEOUT)
        payloads.extend(admission.payloads)
        if admission.metrics is not None:
            return payloads, admission.metrics


async def _lease_is_blocked(buffer: RolloutBuffer) -> bool:
    try:
        await asyncio.wait_for(buffer.acquire_lease(), BLOCKED_TIMEOUT)
    except TimeoutError:
        return True
    return False


@pytest.mark.asyncio
async def test_on_policy_leases_one_batch_per_published_step():
    buffer = _buffer(batch_size=2, max_staleness_steps=0)
    assert await _lease_is_blocked(buffer)

    await buffer.publish(1)
    await _generate(buffer, "a")
    await _generate(buffer, "b")
    assert await _lease_is_blocked(buffer)
    assert (await _take_batch(buffer))[0] == ["a", "b"]
    assert await _lease_is_blocked(buffer)

    await buffer.publish(2)
    lease = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    assert lease.policy_step == 2


@pytest.mark.asyncio
async def test_off_policy_generation_runs_ahead_by_the_staleness_bound():
    buffer = _buffer(batch_size=2, max_in_flight=8, max_staleness_steps=1)
    await buffer.publish(1)
    for uid in "abcd":
        await _generate(buffer, uid)
    assert await _lease_is_blocked(buffer)

    await _take_batch(buffer)
    assert await _lease_is_blocked(buffer)
    await buffer.publish(2)
    for _ in range(2):
        await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    assert await _lease_is_blocked(buffer)


@pytest.mark.asyncio
async def test_leases_never_exceed_max_in_flight():
    buffer = _buffer(batch_size=1, max_in_flight=2, max_staleness_steps=3)
    await buffer.publish(1)
    lease = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    assert await _lease_is_blocked(buffer)

    await _commit(buffer, lease.lease_id, "a")
    assert not await _lease_is_blocked(buffer)


@pytest.mark.asyncio
async def test_surplus_groups_wait_for_the_next_step():
    buffer = _buffer(batch_size=1)
    await buffer.publish(1)
    await _generate(buffer, "a")
    await _generate(buffer, "b")
    assert (await _take_batch(buffer))[0] == ["a"]

    await buffer.publish(2)
    assert (await _take_batch(buffer))[0] == ["b"]


@pytest.mark.asyncio
async def test_groups_stream_to_the_trainer_before_the_batch_completes():
    buffer = _buffer(batch_size=2)
    await buffer.publish(1)
    await _generate(buffer, "a")
    first = await buffer.admit(PROGRESS_TIMEOUT)
    assert (first.payloads, first.metrics) == (["a"], None)

    await _generate(buffer, "b")
    second = await buffer.admit(PROGRESS_TIMEOUT)
    assert second.payloads == ["b"]
    assert second.metrics is not None


@pytest.mark.asyncio
async def test_stale_group_returns_its_prompt_for_regeneration():
    buffer = _buffer(batch_size=1)
    await buffer.publish(1)
    stale = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    for step, uid in enumerate(["a", "b"], start=2):
        await _generate(buffer, uid)
        assert (await _take_batch(buffer))[0] == [uid]
        await buffer.publish(step)

    await _commit(buffer, stale.lease_id, "stale")
    admission = await buffer.admit(PROGRESS_TIMEOUT)
    assert admission.payloads == []
    assert admission.retries == [{"uid": "stale"}]

    await _generate(buffer, "fresh")
    payloads, metrics = await _take_batch(buffer)
    assert payloads == ["fresh"]
    assert metrics["async/rejected_count/stale"] == 1


@pytest.mark.asyncio
async def test_rejected_and_duplicate_groups_are_dropped_and_counted():
    buffer = _buffer(batch_size=2)
    await buffer.publish(1)
    await _generate(buffer, "masked", rejection=AdmissionRejection.FULLY_MASKED)
    await _generate(buffer, "a")
    await _generate(buffer, "a")
    await _generate(buffer, "b")

    payloads, metrics = await _take_batch(buffer)
    assert payloads == ["a", "b"]
    assert metrics["async/rejected_count"] == 2
    assert metrics["async/rejected_count/fully_masked"] == 1
    assert metrics["async/rejected_count/duplicate_uid"] == 1
    assert metrics["async/rejected_rate"] == 0.5


@pytest.mark.asyncio
async def test_dynamic_sampling_filter_discards_uninformative_groups():
    buffer = _buffer(batch_size=1, dynamic_sampling=DynamicSamplingType.FILTER)
    await buffer.publish(1)
    uniform = {"selection": GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD, "rewards": UNIFORM_REWARDS}
    await _generate(buffer, "uniform", **uniform)
    await _generate(buffer, "informative", rewards=SPREAD_REWARDS)

    payloads, metrics = await _take_batch(buffer)
    assert payloads == ["informative"]
    assert metrics["async/dynamic_sampling/candidate_count"] == 2
    assert metrics["async/dynamic_sampling/discarded_count"] == 1
    assert metrics["async/dynamic_sampling/candidate_outcome_reward_mean"] == 0.25
    assert metrics["async/dynamic_sampling/candidate_pass_at_2"] == 0.5


@pytest.mark.asyncio
async def test_dynamic_sampling_fails_when_its_candidate_budget_is_exhausted():
    buffer = _buffer(batch_size=1, dynamic_sampling=DynamicSamplingType.FILTER, max_candidate_groups=2)
    await buffer.publish(1)
    uniform = {"selection": GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD, "rewards": UNIFORM_REWARDS}
    await _generate(buffer, "first", **uniform)
    await _generate(buffer, "second", **uniform)

    with pytest.raises(RuntimeError, match="limit of 2 candidate groups"):
        await buffer.admit(PROGRESS_TIMEOUT)


@pytest.mark.asyncio
async def test_admission_without_progress_raises_a_stall_error():
    buffer = _buffer(batch_size=1)
    await buffer.publish(1)

    with pytest.raises(GroupAdmissionStalledError, match="admitted=0/1 leases=0"):
        await buffer.admit(BLOCKED_TIMEOUT)


@pytest.mark.asyncio
async def test_snapshot_restores_untaken_groups_and_reports_outstanding_leases():
    buffer = _buffer(batch_size=2)
    await buffer.publish(1)
    outstanding = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    await _generate(buffer, "admitted")
    await buffer.admit(PROGRESS_TIMEOUT)
    await _generate(buffer, "extra")

    snapshot = buffer.snapshot()
    assert snapshot.leases == {outstanding.lease_id}

    restored = _buffer(batch_size=2)
    await restored.restore(snapshot)
    await restored.publish(1)
    assert (await _take_batch(restored))[0] == ["admitted", "extra"]


@pytest.mark.asyncio
async def test_every_judged_group_reports_one_disposition_with_its_dwell():
    buffer = _buffer(batch_size=1, dynamic_sampling=DynamicSamplingType.FILTER)
    await buffer.publish(1)
    stale = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    dispositions = []
    for step, uid in enumerate(["a", "b"], start=2):
        await _generate(buffer, uid, rewards=SPREAD_REWARDS)
        while (admission := await buffer.admit(PROGRESS_TIMEOUT)).metrics is None:
            dispositions.extend(admission.dispositions)
        dispositions.extend(admission.dispositions)
        await buffer.publish(step)

    await _commit(buffer, stale.lease_id, "stale", rewards=SPREAD_REWARDS)
    await _generate(buffer, "masked", rejection=AdmissionRejection.FULLY_MASKED)
    uniform = {"selection": GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD, "rewards": UNIFORM_REWARDS}
    await _generate(buffer, "uniform", **uniform)
    await _generate(buffer, "c", rewards=SPREAD_REWARDS)
    while (admission := await buffer.admit(PROGRESS_TIMEOUT)).metrics is None:
        dispositions.extend(admission.dispositions)
    dispositions.extend(admission.dispositions)

    assert [outcome.disposition for outcome in dispositions] == [
        "consumed",
        "consumed",
        "stale",
        "fully_masked",
        "insufficient_reward_spread",
        "consumed",
    ]
    assert all(outcome.tokens == 8 and outcome.dwell_seconds >= 0 for outcome in dispositions)
