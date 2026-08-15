"""Preserve-logprobs-on-soft-timeout routing gate.

FIX 2 for the r5 step-0 stall: a FULLY-GENERATED trajectory whose POST-generation
step failed (VerifierTimeoutError -> no reward, or a soft AgentTimeout without a
verifier result) should KEEP its generated tokens+logprobs at reward=0 instead of
being discarded, so an RLOO-N group is not dropped for all-missing-logprobs.

These tests cover the routing gate ``_should_preserve_timeout_trajectory`` — the
decision to attempt preserve vs. fall back to the discard stub. The AUTHORITATIVE
TIS-validity gate (rollout_logprobs length == response_ids length) runs after
tokenized extraction and is exercised end-to-end by the r6 A/B (needs the full
tokenizer/GPU stack).

Ref: agent_logs/2026-07-03_r5_engine_starvation_rootcause.md
"""

import os
import sys
import types

import pytest

from skyrl_train.utils.harbor_errors import ErrorHandlingConfig

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

# HarborTrajectoryRunner pulls in the harbor/terminal_bench agentic-RL stack, which
# the CPU dev extra deliberately does not install. Skip the module where it is absent
# (it still runs in the agentic RL env where harbor is present).
try:
    from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner  # noqa: E402
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)


def _fake_self(*, preserve: bool = True, classification: bool = True):
    """Minimal stand-in exposing only what the gate helper reads."""
    return types.SimpleNamespace(
        _preserve_logprobs_on_timeout=preserve,
        _error_handling_config=ErrorHandlingConfig(enable_error_classification=classification),
    )


def _result_with(rollout_details):
    agent_result = types.SimpleNamespace(rollout_details=rollout_details)
    return types.SimpleNamespace(agent_result=agent_result)


_gate = HarborTrajectoryRunner._should_preserve_timeout_trajectory


def test_preserve_when_generation_has_logprobs():
    result = _result_with([{"logprobs": [[-0.1, -0.2], [-0.3]]}])
    assert _gate(_fake_self(), result) is True


def test_disabled_flag_falls_back_to_discard():
    result = _result_with([{"logprobs": [[-0.1]]}])
    assert _gate(_fake_self(preserve=False), result) is False


def test_classification_off_is_byte_identical_legacy():
    result = _result_with([{"logprobs": [[-0.1]]}])
    assert _gate(_fake_self(classification=False), result) is False


def test_no_rollout_details_falls_back():
    assert _gate(_fake_self(), _result_with(None)) is False
    assert _gate(_fake_self(), _result_with([])) is False


def test_rollout_details_without_logprobs_falls_back():
    # Generation exists but no logprobs collected -> nothing valid to preserve.
    assert _gate(_fake_self(), _result_with([{"completion_token_ids": [[1, 2]]}])) is False
    assert _gate(_fake_self(), _result_with([{"logprobs": []}])) is False


def test_missing_agent_result_falls_back():
    assert _gate(_fake_self(), types.SimpleNamespace(agent_result=None)) is False
