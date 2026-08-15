"""GAP B real-path: opencode ``version`` + ``opencode_config`` must flow through the
AGENT_SCHEMA -> AgentConfig.kwargs -> OpenCode.__init__ chain.

The correlation-glue smoke passed on CPU mocks while the real path dropped these two
keys: AGENT_SCHEMA exposed ``collect_rollout_details`` but NOT ``version`` /
``opencode_config``, so the opencode 1.18.2 pin + the compaction observable never reached
the agent constructor (opencode installed @latest; gate-4 unexercisable). These tests
drive the REAL builder (HarborConfigBuilder._build_agent_fields) and the REAL harbor
AgentFactory so a future schema regression is caught, not mocked over.
"""

import pytest
from omegaconf import OmegaConf

try:
    from skyrl_train.trajectory_runners.harbor.configuration import AGENT_SCHEMA, HarborConfigBuilder
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)


def _agent_kwargs(harbor_cfg: dict) -> dict:
    cfg = OmegaConf.create({"harbor": harbor_cfg})
    _, kwargs = HarborConfigBuilder(cfg)._build_agent_fields()
    return kwargs


def test_schema_exposes_version_and_opencode_config():
    # Both are kwargs (forwarded to the agent ctor), with NO default so an omitting
    # config forwards nothing (byte-identical for terminus / @latest configs).
    assert "version" in AGENT_SCHEMA.fields
    assert AGENT_SCHEMA.fields["version"].field_type == "kwargs"
    assert AGENT_SCHEMA.fields["version"].default is None
    assert "opencode_config" in AGENT_SCHEMA.fields
    assert AGENT_SCHEMA.fields["opencode_config"].field_type == "kwargs"
    assert AGENT_SCHEMA.fields["opencode_config"].default is None


def test_version_and_opencode_config_forwarded_as_kwargs():
    kwargs = _agent_kwargs(
        {
            "name": "opencode",
            "collect_rollout_details": True,
            "version": "1.18.2",
            "opencode_config": {"compaction": {"auto": True, "reserved": 16384}},
        }
    )
    assert kwargs["version"] == "1.18.2"
    assert kwargs["opencode_config"] == {"compaction": {"auto": True, "reserved": 16384}}
    assert kwargs["collect_rollout_details"] is True


def test_omitting_them_forwards_nothing_byte_identical():
    kwargs = _agent_kwargs({"name": "opencode"})
    assert "version" not in kwargs
    assert "opencode_config" not in kwargs


def test_kwargs_reach_opencode_ctor_via_factory(tmp_path):
    """End-to-end: schema-built kwargs, instantiated through the REAL harbor
    AgentFactory, must land on the OpenCode instance (_version / _opencode_config)."""
    from harbor.agents.factory import AgentFactory
    from harbor.agents.installed.opencode import OpenCode
    from harbor.models.agent.name import AgentName

    kwargs = _agent_kwargs(
        {
            "name": "opencode",
            "collect_rollout_details": True,
            "version": "1.18.2",
            "opencode_config": {"compaction": {"auto": True, "reserved": 16384}},
        }
    )
    # ``name`` is a direct AgentConfig field, not an agent-ctor kwarg.
    kwargs.pop("name", None)
    agent = AgentFactory.create_agent_from_name(
        AgentName.OPENCODE,
        logs_dir=tmp_path,
        model_name="hosted_vllm/my-model",
        **kwargs,
    )
    assert isinstance(agent, OpenCode)
    assert agent._version == "1.18.2"
    assert agent._opencode_config == {"compaction": {"auto": True, "reserved": 16384}}
    # collect_rollout_details mints a per-trial correlation id (the S2 header source).
    assert agent._collect_rollout_details is True
    assert agent._rollout_correlation_id
