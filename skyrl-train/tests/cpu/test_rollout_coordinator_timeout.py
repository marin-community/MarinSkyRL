import asyncio
from types import SimpleNamespace

import pytest

from skyrl_train.trajectory_runners.harbor.rollout_dispatcher import ShardTimeoutPolicy


class LongTailRunner:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    async def run(self, input_batch):
        self.calls += 1
        output = next(self.outputs)
        if output is None:
            await asyncio.Event().wait()
        return output

    async def agent_timeout_output(self, input_batch):
        return {
            "response_ids": [[0]],
            "rollout_metrics": {"generate/errors/AgentTimeoutError": 1},
        }


def _retry_config(
    *,
    max_retries,
    include_exceptions=None,
    exclude_exceptions=None,
    min_wait_sec=0.0,
    max_wait_sec=0.0,
    wait_multiplier=2.0,
):
    return SimpleNamespace(
        max_retries=max_retries,
        include_exceptions=include_exceptions,
        exclude_exceptions=exclude_exceptions,
        min_wait_sec=min_wait_sec,
        max_wait_sec=max_wait_sec,
        wait_multiplier=wait_multiplier,
    )


def test_shard_timeout_default_covers_harbor_attempt_and_backoff_budget():
    retry_config = _retry_config(
        max_retries=3,
        min_wait_sec=60.0,
        max_wait_sec=600.0,
        wait_multiplier=2.0,
    )

    policy = ShardTimeoutPolicy.from_config(
        configured_timeout=None,
        agent_timeout=1800,
        retry_config=retry_config,
    )

    assert policy.timeout_seconds == 7620


def test_shard_timeout_override_is_independent_of_agent_timeout():
    retry_config = _retry_config(max_retries=3)

    policy = ShardTimeoutPolicy.from_config(
        configured_timeout=9000,
        agent_timeout=1800,
        retry_config=retry_config,
    )

    assert policy.timeout_seconds == 9000


@pytest.mark.asyncio
async def test_shard_timeout_retries_when_agent_timeout_is_retryable():
    completed = {"response_ids": [[1]], "rollout_metrics": {}}
    runner = LongTailRunner([None, completed])
    policy = ShardTimeoutPolicy(
        timeout_seconds=0.001,
        retry_config=_retry_config(max_retries=1, include_exceptions={"AgentTimeoutError"}),
    )

    output = await policy.run(runner, {"prompts": ["task"]})

    assert output["response_ids"] == [[1]]
    assert runner.calls == 2
    assert output["rollout_metrics"]["generate/outer_agent_timeouts"] == 1


@pytest.mark.asyncio
async def test_shard_timeout_returns_agent_timeout_when_retry_is_disabled():
    runner = LongTailRunner([None])
    policy = ShardTimeoutPolicy(
        timeout_seconds=0.001,
        retry_config=_retry_config(max_retries=3, exclude_exceptions={"AgentTimeoutError"}),
    )

    output = await policy.run(runner, {"prompts": ["task"]})

    assert runner.calls == 1
    assert output["response_ids"] == [[0]]
    assert output["rollout_metrics"] == {
        "generate/errors/AgentTimeoutError": 1,
        "generate/outer_agent_timeouts": 1,
    }
