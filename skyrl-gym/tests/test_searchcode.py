"""Python and retrieval observations through direct Shellbox task sessions."""

import json

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("code,output", [("print('hello world')", "hello world"), ("print(1 + 1)", "2")])
async def test_python_tool_output_reaches_the_next_model_turn(rollout_session, code, output):
    rollout = await rollout_session(
        "searchcode",
        [f"<tool><python>{code}</python></tool>", "<solution>#### 2</solution>"],
        {"reward_spec": {"ground_truth": "2"}},
    )
    observation = rollout.steps[0].transition.observations[0]
    assert observation == {"role": "user", "content": output}
    assert rollout.steps[1].messages[-2] == observation
    assert rollout.steps[1].transition.reward == 1.0
    assert rollout.loss_mask == (1, 1, 0, 0, 1, 1)


@pytest.mark.asyncio
async def test_python_tool_failure_does_not_end_the_task(rollout_session):
    rollout = await rollout_session(
        "searchcode",
        [
            "<tool><python>raise ValueError('fail')</python></tool>",
            "<tool><python>print(1 + 1)</python></tool>",
            "<solution>#### 2</solution>",
        ],
        {"reward_spec": {"ground_truth": "2"}},
    )
    assert "ValueError: fail" in rollout.steps[0].transition.observations[0]["content"]
    assert rollout.steps[1].transition.observations[0]["content"] == "2"
    assert [step.transition.reward for step in rollout.steps] == [0.0, 0.0, 1.0]
    assert rollout.grade.reward == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_searchcode_sends_a_search_action_to_its_configured_service(rollout_session, retrieval_service):
    url, requests = retrieval_service
    rollout = await rollout_session(
        "searchcode",
        ["<tool><search>France capital</search></tool>", "<solution>#### 2</solution>"],
        {"reward_spec": {"ground_truth": "2"}},
        {"search": {"search_url": url, "log_requests": False}},
    )
    assert requests == [{"query": "France capital", "topk": 3, "return_scores": True}]
    assert json.loads(rollout.steps[0].transition.observations[0]["content"]) == {
        "result": "Doc 1: Paris is the capital of France.\n"
    }
    assert rollout.steps[1].transition.reward == 1.0
