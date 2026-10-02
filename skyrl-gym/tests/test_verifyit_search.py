"""Search tool environments preserve source defaults and delegate final scores."""

import json
import subprocess
import sys

import pytest
from omegaconf import OmegaConf
import skyrl_gym
from skyrl_gym.verification import VerificationStatus


def make_search(route, enabled, truth):
    return skyrl_gym.make(
        route,
        env_config=OmegaConf.create(
            {
                "verifyit_enabled": enabled,
                "search_url": "http://127.0.0.1:1/retrieve",
                "topk": 3,
                "timeout": 1,
                "log_requests": False,
            }
        ),
        extras={"reward_spec": {"ground_truth": truth}, "max_turns": 1},
    )


@pytest.mark.parametrize("route", ["search", "searchcode"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("correct", [False, True])
def test_search_final_factory_step_matches_source_numeric_or_normalized_contract(route, enabled, correct):
    truth = {"target": ["New York", "NYC"]} if route == "search" else "42"
    action = "<answer>The NYC!</answer>" if route == "search" else "<solution>#### 42</solution>"
    if not correct:
        action = "<answer>London</answer>" if route == "search" else "<solution>#### 0</solution>"
    env = make_search(route, enabled, truth)
    try:
        result = env.step(action)
        assert result["done"]
        assert result["reward"] == float(correct)
    finally:
        env.close()


@pytest.mark.parametrize("truth", [{}, {"target": []}, {"target": ["NYC", None]}])
@pytest.mark.parametrize("action", ["<answer>NYC</answer>", "no answer tag"])
def test_search_invalid_trusted_reference_precedes_blank_or_matching_candidate(truth, action):
    env = make_search("search", True, truth)
    try:
        result = env.step(action)
        assert result["reward"] == 0
        assert result["verification"].status is VerificationStatus.ERROR
        assert result["verification"].score is None
        assert result["verification"].diagnostics["verifyit_status"] == "invalid_task"
    finally:
        env.close()


def test_searchcode_invalid_numeric_reference_is_a_minimum_error():
    env = make_search("searchcode", True, None)
    try:
        result = env.step("<solution>#### 42</solution>")
        assert result["reward"] == 0
        assert result["verification"].status is VerificationStatus.ERROR
        assert result["verification"].diagnostics["verifyit_status"] == "invalid_task"
    finally:
        env.close()


@pytest.mark.parametrize("route", ["search", "searchcode"])
@pytest.mark.parametrize("enabled", [False, True])
def test_search_public_step_without_optional_verifyit(route, enabled):
    script = """
import importlib.abc, json, sys
class BlockCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "verifyit" or fullname.startswith("verifyit."):
            raise ModuleNotFoundError("verifyit deliberately unavailable")
sys.meta_path.insert(0, BlockCore())
import skyrl_gym
from omegaconf import OmegaConf
route, enabled = sys.argv[1], sys.argv[2] == "True"
truth = {"target": ["NYC"]} if route == "search" else "42"
config = {"verifyit_enabled": enabled, "search_url": "http://127.0.0.1:1/retrieve", "topk": 3, "timeout": 1, "log_requests": False}
env = skyrl_gym.make(route, env_config=OmegaConf.create(config), extras={"reward_spec": {"ground_truth": truth}, "max_turns": 1})
try:
    result = env.step("<answer>NYC</answer>" if route == "search" else "<solution>#### 42</solution>")
    print(json.dumps({"reward": result["reward"], "status": result["verification"].status.value if "verification" in result else None}))
finally:
    env.close()
"""
    process = subprocess.run(
        [sys.executable, "-c", script, route, str(enabled)], capture_output=True, text=True, timeout=10
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout.splitlines()[-1]) == {
        "reward": 0.0 if enabled else 1.0,
        "status": "error" if enabled else None,
    }


@pytest.mark.parametrize("route", ["search", "searchcode"])
@pytest.mark.parametrize("enabled", [False, True])
def test_shared_reward_boundary_retains_native_exceptions(route, enabled, monkeypatch):
    truth = {"target": ["NYC"]} if route == "search" else "42"
    env = make_search(route, enabled, truth)

    def unavailable(*args):
        raise RuntimeError("source scorer unavailable")

    monkeypatch.setattr(env, "_get_reward", unavailable)
    try:
        if enabled:
            result = env.step("<answer>NYC</answer>" if route == "search" else "<solution>#### 42</solution>")
            assert result["reward"] == 0
            assert result["verification"].status is VerificationStatus.ERROR
            assert result["verification"].diagnostics["verifyit_status"] == "infrastructure_error"
        else:
            with pytest.raises(RuntimeError, match="source scorer unavailable"):
                env.step("<answer>NYC</answer>" if route == "search" else "<solution>#### 42</solution>")
    finally:
        env.close()


@pytest.mark.parametrize("config", [None, {"verifyit_enabled": True}])
def test_nonparticipating_environment_keeps_native_step(config):
    from skyrl_gym.envs.prompt_only.env import PromptOnlyEnv

    env = PromptOnlyEnv(config)
    assert env.step("arbitrary source action") == {"observations": [], "reward": 0.0, "done": True, "metadata": {}}
