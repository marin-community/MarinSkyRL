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
    from skyrl_train.trajectory_runners.harbor.configuration import (
        AGENT_SCHEMA,
        REWARD_SHAPING_SCHEMA,
        HarborConfigBuilder,
        get_exposed_harbor_fields,
    )
    from skyrl_train.trajectory_runners.harbor.identity_aware_reward import IDENTITY_AWARE_SHAPER
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)


def _agent_kwargs(harbor_cfg: dict) -> dict:
    cfg = OmegaConf.create({"harbor": harbor_cfg})
    _, kwargs = HarborConfigBuilder(cfg)._build_agent_fields()
    return kwargs


def test_schema_defaults_are_hygienic():
    assert AGENT_SCHEMA.fields["record_terminal_session"].default is False
    assert AGENT_SCHEMA.fields["trajectory_config"].default == {"raw_content": True}


def test_identity_aware_reward_shaping_is_the_default_strategy():
    assert REWARD_SHAPING_SCHEMA.fields["reward_shaper"].default == IDENTITY_AWARE_SHAPER


def test_omitted_keys_get_defaults():
    kwargs = _agent_kwargs({"name": "terminus-2", "n_concurrent_trials": 8})
    assert kwargs["record_terminal_session"] is False
    assert kwargs["trajectory_config"] == {"raw_content": True}


def test_yaml_can_override_recording_on():
    kwargs = _agent_kwargs({"name": "terminus-2", "record_terminal_session": True})
    assert kwargs["record_terminal_session"] is True


def test_yaml_false_is_honored_no_falsy_bug():
    # The r5 case: explicit `false` must NOT be swallowed by the default.
    kwargs = _agent_kwargs({"name": "terminus-2", "record_terminal_session": False})
    assert kwargs["record_terminal_session"] is False


def test_max_turns_reaches_the_agent_without_deprecated_max_episodes():
    kwargs = _agent_kwargs({"name": "terminus-2", "max_turns": 30})
    assert kwargs["max_turns"] == 30
    assert "max_episodes" not in kwargs


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

    assert retry_config.exclude_exceptions == {"AgentTimeoutError", "VerifierTimeoutError"}


@pytest.mark.parametrize(
    "cfg",
    [
        {"harbor": {"override_timeout_sec": 123}},
        {"override_timeout_sec": 123},
    ],
)
def test_agent_timeout_resolution_supports_nested_and_legacy_layouts(cfg):
    assert HarborConfigBuilder(OmegaConf.create(cfg)).get_agent_timeout_seconds() == 123


def _trial_config(harbor_cfg: dict):
    return HarborConfigBuilder(OmegaConf.create({"harbor": harbor_cfg})).build_trial_config(
        task_path="/tmp/task",
        trials_dir="/tmp/trials",
        model_name="hosted_vllm/model",
        api_base="http://localhost:8000/v1",
        session_id="session",
    )


def test_trial_attempt_timeout_reaches_harbor_trial_config():
    trial_config = _trial_config({"trial_attempt_timeout_sec": 1900})

    assert trial_config.trial_attempt_timeout_sec == 1900
    assert "trial_attempt_timeout_sec" in get_exposed_harbor_fields()["trial"]


def test_trial_attempt_timeout_remains_unset_when_omitted():
    assert _trial_config({}).trial_attempt_timeout_sec is None
