import pytest
from taskcompendium.grading_result import Outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["prompt_only", "preference"])
async def test_prompt_task_returns_terminal_zero_reward(rollout_session, name):
    rollout = await rollout_session(name, ["A model-generated answer"], {"data_source": "example"})
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 0.0)
    assert rollout.steps[0].transition.done is True
    assert rollout.steps[0].transition.observations == ()
    assert rollout.response_token_ids == (20, 21)
