import pytest
from omegaconf import OmegaConf

try:
    from harbor.agents.factory import AgentFactory
    from harbor.agents.installed.pi import Pi
    from harbor.models.agent.name import AgentName
    from skyrl_train.trajectory_runners.harbor.configuration import HarborConfigBuilder
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)


def test_pi_literal_configuration_reaches_the_agent(tmp_path):
    cfg = OmegaConf.create(
        {
            "harbor": {
                "name": "pi",
                "collect_rollout_details": True,
                "thinking_format": "qwen-chat-template",
            }
        }
    )
    _, kwargs = HarborConfigBuilder(cfg)._build_agent_fields()
    kwargs.pop("name", None)

    agent = AgentFactory.create_agent_from_name(
        AgentName.PI,
        logs_dir=tmp_path,
        model_name="hosted_vllm/served-model",
        api_base="https://iris.example/v1",
        **kwargs,
    )

    assert isinstance(agent, Pi)
    assert agent._thinking_format == "qwen-chat-template"
    assert agent._collect_rollout_details is True
    assert agent._rollout_correlation_id
