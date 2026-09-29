"""Agent-specific Harbor YAML keys must reach the real Harbor agent constructor.

A CPU smoke once passed on mocks while the real path dropped OpenCode's ``version`` and
``opencode_config``: AGENT_SCHEMA did not expose them, so the version pin and compaction
settings never reached the agent. These tests build the trial config with the real
builder and instantiate the agent through Harbor's real AgentFactory.
"""

import pytest
from omegaconf import OmegaConf

try:
    from harbor.agents.factory import AgentFactory
    from harbor.agents.installed.opencode import OpenCode
    from harbor.agents.installed.pi import Pi
    from skyrl_train.trajectory_runners.harbor.configuration import HarborConfigBuilder
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)


def _agent_config(harbor_cfg: dict):
    return (
        HarborConfigBuilder(OmegaConf.create({"harbor": harbor_cfg}))
        .build_trial_config(
            task_path="/tmp/task",
            trials_dir="/tmp/trials",
            model_name="hosted_vllm/served-model",
            api_base="https://iris.example/v1",
            session_id="session",
        )
        .agent
    )


@pytest.mark.parametrize(
    ("harbor_cfg", "agent_type", "expected_attributes"),
    [
        (
            {
                "name": "opencode",
                "collect_rollout_details": True,
                "version": "1.18.2",
                "opencode_config": {"compaction": {"auto": True, "reserved": 16384}},
            },
            OpenCode,
            {"_version": "1.18.2", "_opencode_config": {"compaction": {"auto": True, "reserved": 16384}}},
        ),
        (
            {"name": "pi", "collect_rollout_details": True, "thinking_format": "qwen-chat-template"},
            Pi,
            {"_thinking_format": "qwen-chat-template"},
        ),
    ],
    ids=["opencode", "pi"],
)
def test_agent_configuration_reaches_the_harbor_agent(tmp_path, harbor_cfg, agent_type, expected_attributes):
    agent = AgentFactory.create_agent_from_config(_agent_config(harbor_cfg), logs_dir=tmp_path)

    assert isinstance(agent, agent_type)
    # Harbor agents expose their constructor settings only as private attributes.
    assert {name: getattr(agent, name) for name in expected_attributes} == expected_attributes
    # collect_rollout_details mints the per-trial correlation id sent as a request header.
    assert agent._collect_rollout_details is True
    assert agent._rollout_correlation_id


def test_omitted_opencode_settings_are_not_forwarded():
    kwargs = _agent_config({"name": "opencode"}).kwargs

    assert "version" not in kwargs
    assert "opencode_config" not in kwargs
