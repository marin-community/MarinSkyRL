"""Public source clients preserve extraction, partial scores and failure boundaries."""

import json

import pytest
from taskcompendium.grading_result import Outcome
from skyrl_gym.answer_tasks import grade_reasoning_gym
from skyrl_gym.envs.nemotron_ultra.mcqa import grade_mcqa


@pytest.mark.parametrize("response", ["Answer: 42", "Answer: wrong\nAnswer: x = 42", "42"])
def test_reasoning_client_preserves_native_fractional_score(model_turn, response):
    entry = {"answer": "42", "metadata": {"source_dataset": "simple_equations"}}
    extras = {"reward_model": {"ground_truth": {"task": "simple_equations", "entry": entry}}}
    native = grade_reasoning_gym(model_turn(response), {}, extras)
    client = grade_reasoning_gym(model_turn(response), {"verifyit_enabled": True}, extras)
    assert client.reward == native.reward


@pytest.mark.parametrize(
    "mode,response",
    [
        ("strict_single_letter_boxed", r"\boxed{Z}"),
        ("lenient_boxed", r"\boxed{zebra}"),
        ("lenient_answer_colon", "Answer: zebra"),
        ("lenient_answer_colon_md", "**Answer:** Z"),
    ],
)
def test_mcqa_client_preserves_source_extraction_noncontiguous_options(mode, response):
    record = {"options": [{"A": "apple"}, {"Z": "zebra"}], "expected_answer": "Z", "grading_mode": mode}
    assert grade_mcqa(response, record, verifyit_enabled=True) == grade_mcqa(response, record)
    assert grade_mcqa(response, record, verifyit_enabled=True)[0] == 1.0
    assert grade_mcqa(response.replace("Z", "B").replace("zebra", "apple"), record, verifyit_enabled=True)[0] == 0


@pytest.mark.asyncio
async def test_invalid_reference_does_not_become_successful_verification(rollout_session):
    record = {"options": [{"A": "apple"}], "expected_answer": "Z"}
    extras = {
        "extra_info": {
            "nemotron_ultra": {
                "route": "task_session",
                "agent": "mcqa_simple_agent",
                "record_json": json.dumps(record),
                "request_json": "{}",
            }
        }
    }
    rollout = await rollout_session("nemotron_ultra", [r"\boxed{Z}"], extras, {"verifyit_enabled": True})
    assert rollout.steps[0].transition.reward == 0
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.INFRA_ERROR, None)
