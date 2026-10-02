"""Public source clients preserve extraction, partial scores and failure boundaries."""

import base64
import json
import pickle
import subprocess
import sys

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
    assert grade_mcqa(response, record, verifyit_enabled=True)[0] == grade_mcqa(response, record)[0]
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


@pytest.mark.parametrize("candidate", [r"\boxed{Z}", ""])
@pytest.mark.parametrize(
    "change",
    [
        {"expected_answer": "Q"},
        {"options": [1]},
        {"grading_mode": "misspelled"},
        {"template_metadata": {"output_regex": "("}},
        {"template_metadata": {"output_regex": "a{99999999999999999999999999}"}},
    ],
)
def test_mcqa_trusted_failures_are_invalid_tasks_before_candidate_extraction(change, candidate):
    record = {"options": [{"A": "apple"}, {"Z": "zebra"}], "expected_answer": "Z", **change}
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
    try:
        result = env.step(candidate)
        assert result["verification"].status is VerificationStatus.ERROR
        assert result["verification"].score is None
        assert result["metadata"]["verifyit_status"] == "invalid_task"
    finally:
        env.close()


def test_mcqa_regex_deadline_is_terminal_and_next_grade_succeeds():
    from skyrl_gym.envs.mcq.verifyit import MCQPolicy, grade_mcq

    record = {"options": [{"A": "apple"}], "expected_answer": "A", "template_metadata": {"output_regex": "(a+)+$"}}
    reward, details = grade_mcq("a" * 100000 + "!", record, policy=MCQPolicy.ULTRA, timeout=1.0)
    assert reward == 0
    assert details["verifyit_status"] == "infra_error"
    assert grade_mcq(r"\boxed{A}", {**record, "template_metadata": {}}, policy=MCQPolicy.ULTRA)[0] == 1


def test_mcq_raw_capture_and_source_policy_preserve_distinct_box_selection():
    from skyrl_gym.envs.mcq.verifyit import MCQPolicy, grade_mcq, structure_mcq

    record = {"options": [{"A": "apple"}, {"Z": "zebra"}], "expected_answer": "Z"}
    response = r"<think>\boxed{A}</think>\boxed{Z}"
    captured = structure_mcq(response, record)
    record["options"][0]["A"] = "changed"
    assert captured.candidate == response
    assert captured.record["options"][0]["A"] == "apple"
    assert grade_mcq(response, captured.record, policy=MCQPolicy.ULTRA)[0] == 1
    assert grade_mcq(response, captured.record, policy=MCQPolicy.FIRST_BOX)[0] == 0


@pytest.mark.parametrize("route", ["mcq", "nemotron_ultra"])
@pytest.mark.parametrize(
    "worker_failure", ["missing_interpreter", "failed_worker", "unreadable_worker", "worker_exception"]
)
def test_mcq_worker_infrastructure_failures_return_unscored_minimum(route, worker_failure, monkeypatch, tmp_path):
    if route == "mcq":
        extras = {"reward_model": {"ground_truth": "A"}}
    else:
        extras = {
            "extra_info": {
                "nemotron_ultra": {
                    "route": "skyrl_gym",
                    "agent": "mcqa_simple_agent",
                    "record_json": json.dumps({"options": [{"A": "apple"}], "expected_answer": "A"}),
                    "request_json": "{}",
                }
            }
        }
    executable = tmp_path / "worker"
    if worker_failure != "missing_interpreter":
        payload = base64.b64encode(pickle.dumps((False, TypeError("worker grading failure")))).decode()
        commands = {
            "failed_worker": "exit 1\n",
            "unreadable_worker": "echo not-a-result\n",
            "worker_exception": f"echo {payload}\n",
        }
        executable.write_text("#!/bin/sh\n" + commands[worker_failure])
        executable.chmod(0o700)
    env = skyrl_gym.make(route, env_config=OmegaConf.create({"verifyit_enabled": True}), extras=extras)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(sys, "executable", str(executable))
            result = env.step(r"\boxed{A}")
        assert result["reward"] == 0
        assert result["verification"].status is VerificationStatus.ERROR
        assert result["verification"].score is None
        assert result["metadata"]["verifyit_status"] == "infra_error"
    finally:
        env.close()


def test_mcq_missing_dependency_is_typed_on_enabled_routes_and_native_still_grades():
    program = r"""
import importlib.abc
import base64
import json
import pickle
import sys
class MissingVerifyit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "verifyit" or fullname.startswith("verifyit."):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, MissingVerifyit())
from omegaconf import OmegaConf
import skyrl_gym
results = []
for route, extras in [
    ("mcq", {"reward_model": {"ground_truth": "A"}}),
    ("nemotron_ultra", {"extra_info": {"nemotron_ultra": {
        "route": "skyrl_gym", "agent": "mcqa_simple_agent",
        "record_json": json.dumps({"options": [{"A": "apple"}], "expected_answer": "A"}),
        "request_json": "{}",
    }}}),
]:
    for enabled in [False, True]:
        env = skyrl_gym.make(route, env_config=OmegaConf.create({"verifyit_enabled": enabled}), extras=extras)
        try:
            result = env.step(r"\boxed{A}")
            results.append([result["reward"], result["metadata"].get("verifyit_status")])
        finally:
            env.close()
print(json.dumps(results))
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [[1.0, None], [0.0, "infra_error"], [1.0, None], [0.0, "infra_error"]]
