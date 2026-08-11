"""CPU unit tests for the single-turn delphi math-RLVR launcher path.

Pins the `configs/delphi_math_rl.yaml` contract and the TP-divides-heads guard so a
future edit that drops the `environment` flatten, the parquet data-kind routing, the 4k
cap, or the TP-42 guard fails here instead of silently at rollout time on a GPU gang.

Run:
    python -m pytest cloud/iris/tests/test_delphi_math_rl_config.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.rl_config_translation import (  # noqa: E402
    build_skyrl_hydra_args,
    parse_rl_config,
    validate_tp_divides_heads,
)

_CONFIG = "cloud/iris/configs/delphi_math_rl.yaml"


@dataclass
class _HPCStub:
    gpus_per_node: int = 8


def test_delphi_config_parses_to_main_base_non_agentic():
    parsed = parse_rl_config(_CONFIG)
    assert parsed.entrypoint == "skyrl_train.entrypoints.main_base"
    # Non-agentic: no terminal_bench / teacher sections.
    assert parsed.terminal_bench is None
    assert parsed.teacher is None
    # env_class routed via the `environment` section (default_env_class fallback).
    assert parsed.environment.get("env_class") == "aime"
    # Parquet data kind is popped out of `data` (must not reach Hydra).
    assert parsed.data_kind == "parquet"
    assert "kind" not in parsed.data
    # 4k cap + TP=2 (divides 42).
    assert parsed.tensor_parallel_size == 2
    assert parsed.generator["engine_init_kwargs"]["max_model_len"] == 4096
    # The chat-template override + head-count guard keys are declared.
    assert parsed.raw["policy_chat_template"].endswith("delphi_v0.jinja2")
    assert parsed.raw["model_num_attention_heads"] == 42


def test_delphi_config_flattens_environment_and_caps_into_hydra_args():
    parsed = parse_rl_config(_CONFIG)
    exp_args = {"job_name": "delphi-math-rl-test", "experiments_dir": "/tmp/exp", "num_nodes": 4}
    args = build_skyrl_hydra_args(parsed, exp_args, _HPCStub())

    # environment.env_class must be flattened (regression guard: the flatten loop
    # historically covered only trainer/generator/data).
    assert "environment.env_class=aime" in args
    # engine_init_kwargs is an "optional" (++) section, so the key carries a ++ prefix.
    assert any(a.endswith("generator.engine_init_kwargs.max_model_len=4096") for a in args)
    assert "trainer.algorithm.advantage_estimator=grpo" in args
    # Launcher-only keys must NOT leak into Hydra.
    assert not any("data.kind" in a for a in args)
    assert not any("policy_chat_template" in a for a in args)
    assert not any("model_num_attention_heads" in a for a in args)


def test_iris_derives_durable_training_trajectory_path():
    parsed = parse_rl_config(_CONFIG)
    args = build_skyrl_hydra_args(
        parsed,
        {"job_name": "retained-run", "experiments_dir": "s3://bucket/iris/", "num_nodes": 4},
        _HPCStub(),
    )

    assert (
        "generator.trajectory_retention.output_path='s3://bucket/iris/retained-run/trace_jobs/training_trajectories'"
        in args
    )


def test_model_source_locator_reaches_trainer_config():
    parsed = parse_rl_config(_CONFIG)
    exp_args = {
        "job_name": "exportable-run",
        "model_path": "/tmp/materialized-model",
        "model_source_uri": "s3://models/policy",
        "model_source_identity": "policy@abc123",
        "num_nodes": 4,
    }

    args = build_skyrl_hydra_args(parsed, exp_args, _HPCStub())

    assert "trainer.policy.model.source_uri='s3://models/policy'" in args
    assert "trainer.policy.model.source_identity='policy@abc123'" in args


def test_partial_model_source_is_rejected_during_config_translation():
    parsed = parse_rl_config(_CONFIG)
    exp_args = {
        "job_name": "invalid-exportable-run",
        "model_path": "/tmp/materialized-model",
        "model_source_identity": "policy@abc123",
        "num_nodes": 4,
    }

    with pytest.raises(ValueError, match="must be provided together"):
        build_skyrl_hydra_args(parsed, exp_args, _HPCStub())


def test_tp_guard_rejects_tp8_on_42_heads():
    with pytest.raises(ValueError, match="does not divide"):
        validate_tp_divides_heads(8, 42)


@pytest.mark.parametrize("tp", [1, 2, 3, 6, 7, 14, 21, 42])
def test_tp_guard_allows_divisors_of_42(tp):
    validate_tp_divides_heads(tp, 42)  # must not raise


def test_tp_guard_noop_when_heads_unset():
    validate_tp_divides_heads(8, None)  # existing configs (no head count) are unaffected


def test_parse_rejects_bad_tp_against_declared_heads(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "entrypoint: skyrl_train.entrypoints.main_base\n"
        "model_num_attention_heads: 42\n"
        "context_budget:\n"
        "  request_window_tokens: 4096\n"
        "  max_new_tokens_per_turn: 3584\n"
        "  max_turns: 1\n"
        "generator:\n"
        "  inference_engine_tensor_parallel_size: 8\n"
    )
    with pytest.raises(ValueError, match="does not divide"):
        parse_rl_config(str(bad))
