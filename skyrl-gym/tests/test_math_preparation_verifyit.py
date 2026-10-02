import hashlib
import json
import os
import subprocess
import sys
import threading

import pytest
from omegaconf import DictConfig

import skyrl_gym
from skyrl_gym.envs.math_verifyit import MathPolicy, structure_math
from skyrl_gym.verification import RolloutEvidence


def make_env(route, reference, **config):
    return skyrl_gym.make(
        route,
        env_config=DictConfig({"verifyit_enabled": True, **config}),
        extras={
            "reward_model": {"ground_truth": reference},
            "reward_spec": {"ground_truth": reference},
            "max_turns": 2,
        },
    )


def test_structural_capture_keeps_null_and_literal_fields():
    captured = structure_math(None, " 0042 ", "length")
    assert captured.response is None
    assert captured.reference == " 0042 "
    assert captured.stop_reason == "length"


@pytest.mark.parametrize(
    "method,expected,policy",
    [
        ("strict", 1, MathPolicy.GSM_STRICT),
        ("flexible", 0, MathPolicy.GSM_FLEXIBLE),
        ("final_line", 0, MathPolicy.GSM_COMPLETED_FINAL_LINE),
    ],
)
def test_source_policy_retains_original_response(method, expected, policy):
    response = "Work: #### $42\nCorrection: #### 43"
    env = make_env("gsm8k", "42", reward_method=method)
    env.set_rollout_evidence(RolloutEvidence(response=response, stop_reason="stop"))
    result = env.step(response)
    assert result["reward"] == expected
    preparation = result["verification"].diagnostics["preparation"]
    raw = {"response": response, "reference": "42", "stop_reason": "stop"}
    assert preparation["raw_sha256"] == hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    assert "raw" not in preparation
    assert "expected" not in result["verification"].diagnostics
    assert preparation["policy"] == policy


@pytest.mark.parametrize("response,reward", [(None, -1), ("no answer", -1), (r"\boxed{}", 1)])
def test_boxed_missing_answer_is_distinct_from_empty_answer(response, reward):
    result = make_env("aime", "", strict_box_verify=True).step(response)
    assert result["reward"] == reward
    assert result["verification"].diagnostics["verifyit_status"] == "scored"


@pytest.mark.parametrize("route,minimum", [("aime", -1), ("gsm8k", 0), ("gsm8k_multi_turn", 0)])
def test_deadline_and_ordinary_preparation_failure_cannot_earn_credit(route, minimum):
    response = "Answer: 42\n#### 42"
    timed = make_env(route, "42", verifyit_timeout=0.000001).step(response)
    malformed = make_env(route, "42").step([response])
    for result in (timed, malformed):
        assert result["reward"] == minimum
        assert result["done"]
        assert result["verification"].status.value == "error"
        assert result["verification"].diagnostics["verifyit_status"] == "infra_error"
        assert "reward_result" not in result


@pytest.mark.parametrize("route,minimum", [("aime", -1), ("gsm8k", 0), ("gsm8k_multi_turn", 0)])
def test_invalid_reference_precedes_missing_answer(route, minimum):
    result = make_env(route, None).step(None)
    assert result["reward"] == minimum
    assert result["verification"].diagnostics["verifyit_status"] == "invalid_task"


def test_multiturn_format_bonus_is_separate_from_correctness():
    result = make_env("gsm8k_multi_turn", "42").step("#### 41")
    assert result["reward"] == 0.1
    assert result["verification"].score == 0
    assert result["verification"].passed is False
    assert result["reward_result"].components == {"format": 0.1}
    assert not result["done"]


@pytest.mark.parametrize("route,minimum", [("aime", -1), ("gsm8k", 0), ("gsm8k_multi_turn", 0)])
def test_semantically_invalid_reference_precedes_malformed_candidate(route, minimum):
    result = make_env(route, "nan").step(["42"])
    assert result["reward"] == minimum
    assert result["verification"].diagnostics["verifyit_status"] == "invalid_task"


@pytest.mark.parametrize("candidate,reward", [("(18,-24)", 1.0), ("(-24,18)", -1.0), ("(18.0,-24)", -1.0)])
def test_aime_tuple_source_spelling_through_registered_route(candidate, reward):
    result = make_env("aime", "(18,-24)").step(f"Answer: {candidate}")
    assert result["reward"] == reward
    assert result["verification"].status.value == "verified"


@pytest.mark.parametrize("reference", ["(1e309,2)", "(-1e309,2)"])
def test_aime_nonfinite_tuple_reference_is_not_a_positive_literal_match(reference):
    result = make_env("aime", reference).step(f"Answer: {reference}")
    assert result["reward"] == -1
    assert result["verification"].diagnostics["verifyit_status"] == "invalid_task"


def test_aime_total_deadline_has_one_owned_worker_and_reaps_it():
    owned = {os.getpid()}
    observed = []
    stopped = threading.Event()

    def workers():
        rows = subprocess.check_output(["ps", "-ww", "-eo", "pid,ppid,command"], text=True).splitlines()
        candidates = [row.split(None, 2) for row in rows if " -m verifyit.execution.worker" in row]
        for pid, parent, _ in candidates:
            if int(parent) in owned:
                owned.add(int(pid))
        return {int(pid) for pid, _, _ in candidates if int(pid) in owned}

    def monitor():
        while not stopped.is_set():
            observed.append(workers())
            stopped.wait(0.02)

    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        result = make_env("aime", "42", verifyit_timeout=3.0).step("Answer: 42")
        remaining = workers()
    finally:
        stopped.set()
        thread.join()
    assert observed and max(map(len, observed)) == 1
    assert not remaining
    if result["verification"].status.value == "error":
        assert result["reward"] == -1
        assert result["verification"].diagnostics["verifyit_status"] == "infra_error"
    else:
        assert result["reward"] == 1


def test_enabled_math_missing_dependency_returns_minimum_error():
    program = r"""
import importlib.abc
import sys

class BlockVerifyit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "verifyit" or fullname.startswith("verifyit."):
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockVerifyit())
import skyrl_gym
from omegaconf import DictConfig

for route, minimum in [("aime", -1), ("gsm8k", 0), ("gsm8k_multi_turn", 0)]:
    env = skyrl_gym.make(
        route, env_config=DictConfig({"verifyit_enabled": True}),
        extras={"reward_model": {"ground_truth": "42"},
                "reward_spec": {"ground_truth": "42"}, "max_turns": 2},
    )
    result = env.step("Answer: 42\n#### 42")
    assert result["reward"] == minimum
    assert result["done"]
    assert result["verification"].status.value == "error"
    assert result["verification"].diagnostics["verifyit_status"] == "infra_error"
"""
    subprocess.run(
        [sys.executable, "-c", program], env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}, check=True
    )
