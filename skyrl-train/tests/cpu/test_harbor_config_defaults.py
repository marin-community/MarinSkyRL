"""Config-hygiene DEFAULTS for the harbor terminus-2 agent config.

These assert that a terminal_bench yaml which OMITS the hygiene keys still gets
the safe RL defaults (recording off, raw trajectory content on), and that an
explicit yaml value still OVERRIDES the default in both directions (no falsy
`or default` bug that would silently re-enable recording).

Regression guard for the r5 engine-starvation investigation
(agent_logs/2026-07-03_r5_engine_starvation_rootcause.md).
"""

import os
import sys

import pytest
from omegaconf import OmegaConf

# The builder lives under examples/ (not an installed package).
_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

# The builder pulls in the harbor/terminal_bench agentic-RL stack, which the CPU
# dev extra deliberately does not install. Skip the module where it is absent
# (it still runs in the agentic RL env where harbor is present).
try:
    from terminal_bench.harbor_config import AGENT_SCHEMA, HarborConfigBuilder  # noqa: E402
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)


def _agent_kwargs(harbor_cfg: dict) -> dict:
    cfg = OmegaConf.create({"harbor": harbor_cfg})
    _, kwargs = HarborConfigBuilder(cfg)._build_agent_fields()
    return kwargs


def test_schema_defaults_are_hygienic():
    assert AGENT_SCHEMA.fields["record_terminal_session"].default is False
    assert AGENT_SCHEMA.fields["trajectory_config"].default == {"raw_content": True}


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
