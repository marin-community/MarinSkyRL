"""Public source clients preserve extraction, partial scores and failure boundaries."""

import json

import pytest
from omegaconf import OmegaConf

import skyrl_gym
from skyrl_gym.envs.nemotron_ultra.mcqa import grade_mcqa
from skyrl_gym.verification import VerificationStatus


@pytest.mark.parametrize("response", ["Answer: 42", "Answer: wrong\nAnswer: x = 42", "42"])
def test_reasoning_client_preserves_native_fractional_score(response):
    entry = {"answer": "42", "metadata": {"source_dataset": "simple_equations"}}
    extras = {"reward_model": {"ground_truth": {"task": "simple_equations", "entry": entry}}}
    native = skyrl_gym.make("reasoning_gym", env_config=OmegaConf.create({}), extras=extras)
    client = skyrl_gym.make("reasoning_gym", env_config=OmegaConf.create({"verifyit_enabled": True}), extras=extras)
    assert client.step(response)["reward"] == native.step(response)["reward"]


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


def test_invalid_reference_does_not_become_successful_verification():
    record = {"options": [{"A": "apple"}], "expected_answer": "Z"}
    extras = {
        "extra_info": {
            "nemotron_ultra": {
                "route": "skyrl_gym",
                "agent": "mcqa_simple_agent",
                "record_json": json.dumps(record),
                "request_json": "{}",
            }
        }
    }
    env = skyrl_gym.make("nemotron_ultra", env_config=OmegaConf.create({"verifyit_enabled": True}), extras=extras)
    result = env.step(r"\boxed{Z}")
    assert result["reward"] == 0
    assert result["verification"].score is None


@pytest.mark.parametrize("route", ["reasoning_gym", "nemotron_ultra"])
@pytest.mark.parametrize("defect", ["duplicate_answer", "nonfinite_metadata"])
@pytest.mark.parametrize("candidate", ["Answer: 42", ""])
def test_reasoning_trusted_json_defect_is_an_error_even_for_blank_candidates(route, defect, candidate):
    record = {"question": "What is 6 times 7?", "answer": "42", "metadata": {"source_dataset": "simple_equations"}}
    if defect == "nonfinite_metadata":
        record["metadata"]["unexpected"] = float("nan")
    encoded = json.dumps(record)
    if defect == "duplicate_answer":
        encoded = encoded.replace('"answer": "42"', '"answer": "wrong", "answer": "42"')
    if route == "reasoning_gym":
        extras = {"reward_model": {"ground_truth": '{"task":"simple_equations","entry":' + encoded + "}"}}
    else:
        extras = {
            "extra_info": {
                "nemotron_ultra": {
                    "route": "skyrl_gym",
                    "agent": "reasoning_gym_simple_agent",
                    "record_json": encoded,
                    "request_json": "{}",
                }
            }
        }
    env = skyrl_gym.make(route, env_config=OmegaConf.create({"verifyit_enabled": True}), extras=extras)
    try:
        result = env.step(candidate)
        assert result["reward"] == 0
        assert result["verification"].status is VerificationStatus.ERROR
        assert result["verification"].score is None
    finally:
        env.close()
    # Preserve the source path's permissive JSON behavior, even though opt-in rejects it.
    native = skyrl_gym.make(route, env_config=OmegaConf.create({}), extras=extras)
    try:
        assert native.step("Answer: 42")["reward"] == 1
    finally:
        native.close()
