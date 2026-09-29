"""Config-hygiene DEFAULTS for the harbor terminus-2 agent config.

These assert that a terminal_bench yaml which OMITS the hygiene keys still gets
the safe RL defaults (recording off, raw trajectory content on), and that an
explicit yaml value still OVERRIDES the default in both directions (no falsy
`or default` bug that would silently re-enable recording).

Regression guard for the r5 engine-starvation investigation
(agent_logs/2026-07-03_r5_engine_starvation_rootcause.md).
"""

import pytest
from omegaconf import OmegaConf

# The trainer CPU gate installs Harbor through the dedicated harbor-test group.
# Keep the import guard for minimal launcher environments that collect this file.
try:
    from skyrl_train.trajectory_runners.harbor.configuration import HarborConfigBuilder
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)


def _trial_config(harbor_cfg: dict):
    return HarborConfigBuilder(OmegaConf.create({"harbor": harbor_cfg})).build_trial_config(
        task_path="/tmp/task",
        trials_dir="/tmp/trials",
        model_name="hosted_vllm/model",
        api_base="http://localhost:8000/v1",
        session_id="session",
    )


def _agent_kwargs(harbor_cfg: dict) -> dict:
    return _trial_config(harbor_cfg).agent.kwargs


def test_omitted_keys_get_defaults():
    kwargs = _agent_kwargs({"name": "terminus-2", "n_concurrent_trials": 8})
    assert kwargs["record_terminal_session"] is False
    assert kwargs["trajectory_config"] == {"raw_content": True}


# An explicit `false` must not be swallowed by the default.
@pytest.mark.parametrize("record_terminal_session", [True, False])
def test_explicit_record_terminal_session_is_honored(record_terminal_session):
    kwargs = _agent_kwargs({"name": "terminus-2", "record_terminal_session": record_terminal_session})
    assert kwargs["record_terminal_session"] is record_terminal_session


def test_max_turns_reaches_the_agent_without_deprecated_max_episodes():
    kwargs = _agent_kwargs({"name": "terminus-2", "max_turns": 30})
    assert kwargs["max_turns"] == 30
    assert "max_episodes" not in kwargs


def test_llm_call_kwargs_reach_the_agent():
    kwargs = _agent_kwargs({"name": "terminus-2", "llm_call_kwargs": {"max_tokens": 4096}})

    assert kwargs["llm_call_kwargs"] == {"max_tokens": 4096}


def test_passthrough_exceptions_are_never_retried():
    cfg = OmegaConf.create(
        {
            "harbor": {
                "passthrough_exceptions": ["AgentTimeoutError"],
                "exclude_exceptions": ["VerifierTimeoutError"],
            }
        }
    )

    retry_config = HarborConfigBuilder(cfg).build_retry_config()

    assert {
        "AgentTimeoutError",
        "OutputLengthExceededError",
        "TurnCapExhaustedError",
        "VerifierTimeoutError",
    } <= retry_config.exclude_exceptions


@pytest.mark.parametrize(
    "cfg",
    [
        {"harbor": {"override_timeout_sec": 123}},
        {"override_timeout_sec": 123},
    ],
)
def test_agent_timeout_resolution_supports_nested_and_legacy_layouts(cfg):
    assert HarborConfigBuilder(OmegaConf.create(cfg)).get_agent_timeout_seconds() == 123


def test_trial_attempt_timeout_reaches_harbor_trial_config():
    trial_config = _trial_config({"trial_attempt_timeout_sec": 1900})

    assert trial_config.trial_attempt_timeout_sec == 1900


def test_daytona_ttl_reaches_harbor_environment_config():
    trial_config = _trial_config({"ttl_minutes": 90})

    assert trial_config.environment.kwargs["ttl_minutes"] == 90


def test_daytona_network_policy_reaches_harbor_environment_config():
    policy = {"mode": "domain_allow_list", "value": "iris.oa.dev"}

    trial_config = _trial_config({"env_network_policy": policy})

    assert trial_config.environment.kwargs["network_policy"] == policy
