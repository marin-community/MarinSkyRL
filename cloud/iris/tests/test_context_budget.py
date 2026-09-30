"""Behavioral tests for the Iris RL context-budget contract."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import fsspec
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.rl_config_translation import (  # noqa: E402
    ContextBudget,
    compose_skyrl_config,
    parse_rl_config,
    write_resolved_context_budget,
)


@dataclass
class _HPCStub:
    gpus_per_node: int = 8


@pytest.mark.parametrize(
    "config_path", sorted((_REPO_ROOT / "cloud/iris/configs").glob("*.yaml")), ids=lambda p: p.name
)
def test_iris_config_materializes_one_coherent_context_budget(config_path):
    parsed = parse_rl_config(str(config_path))
    budget = parsed.context_budget
    window = budget.request_window_tokens
    output = budget.max_new_tokens_per_turn

    assert parsed.trainer["max_prompt_length"] + parsed.generator["sampling_params"]["max_generate_length"] == window
    assert parsed.generator["max_input_length"] == budget.max_input_tokens
    assert parsed.generator["engine_init_kwargs"]["max_model_len"] == window
    assert parsed.generator["max_turns"] == budget.max_turns
    if parsed.terminal_bench is not None:
        assert parsed.terminal_bench["harbor"]["max_turns"] == budget.max_turns
        assert parsed.terminal_bench["harbor"]["llm_call_kwargs"]["max_tokens"] == output
        assert parsed.terminal_bench["model_info"] == {
            "max_input_tokens": budget.max_input_tokens,
            "max_output_tokens": output,
        }


@pytest.mark.parametrize(
    "config_path", sorted((_REPO_ROOT / "cloud/iris/configs").glob("snowball_ultra_*.yaml")), ids=lambda p: p.name
)
def test_snowball_ultra_judges_read_credentials_from_the_environment(config_path):
    ultra = yaml.safe_load(config_path.read_text())["environment"]["skyrl_gym"]["nemotron_ultra"]

    for judge in (ultra["judges"]["general"], ultra["judges"]["safety"], ultra["genrm"]["judge"]):
        assert judge["api_key_env"]
        assert "api_key" not in judge


def test_taskcompendium_launch_limits_reach_the_composed_runtime(tmp_path):
    config = tmp_path / "taskcompendium.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "entrypoint": "taskcompendium",
                "config_groups": {"taskcompendium_config": "taskcompendium"},
                "context_budget": {
                    "request_window_tokens": 8192,
                    "max_new_tokens_per_turn": 256,
                    "max_turns": 3,
                },
                "taskcompendium": {
                    "concurrency": 2,
                    "max_turns": 3,
                    "timeout": 300,
                    "parallel_tool_calls": False,
                },
            }
        )
    )
    parsed = parse_rl_config(str(config))
    cfg = compose_skyrl_config(parsed, {"num_nodes": 2}, _HPCStub()).config

    assert dict(cfg.taskcompendium_config) == {
        "concurrency": 2,
        "max_turns": 3,
        "timeout": 300,
        "parallel_tool_calls": False,
    }
    assert cfg.taskcompendium_config.max_turns == cfg.generator.max_turns


def test_context_budget_derives_all_hydra_length_arguments():
    parsed = parse_rl_config(str(_REPO_ROOT / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"))
    cfg = compose_skyrl_config(parsed, {"job_name": "context-test", "num_nodes": 4}, _HPCStub()).config

    assert cfg.trainer.max_prompt_length == 114688
    assert cfg.generator.max_input_length == 114688
    assert cfg.generator.max_turns == 90
    assert cfg.generator.sampling_params.max_generate_length == 16384
    assert cfg.generator.engine_init_kwargs.max_model_len == 131072
    assert cfg.terminal_bench_config.model_info.max_input_tokens == 114688
    assert cfg.terminal_bench_config.model_info.max_output_tokens == 16384
    assert cfg.terminal_bench_config.harbor.max_turns == 90
    assert cfg.terminal_bench_config.harbor.llm_call_kwargs.max_tokens == 16384
    assert cfg.generator.trajectory_reward_shaping.overlong.l_max == 65536
    assert cfg.generator.trajectory_reward_shaping.overlong.l_cache == 16384


@pytest.mark.parametrize(
    ("max_turns", "expected_l_max", "expected_l_cache"),
    [(1, 4096, 1024), (30, 16384, 4096)],
)
def test_context_budget_derives_overlong_window_for_single_and_multi_turn(
    tmp_path, max_turns, expected_l_max, expected_l_cache
):
    config = tmp_path / "overlong.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "context_budget": {
                    "request_window_tokens": 32768,
                    "max_new_tokens_per_turn": 4096,
                    "max_turns": max_turns,
                },
                "generator": {"trajectory_reward_shaping": {"enabled": True}},
            }
        )
    )

    parsed = parse_rl_config(str(config))

    assert parsed.context_budget.generated_tokens_per_trajectory == expected_l_max
    assert parsed.generator["trajectory_reward_shaping"]["overlong"] == {
        "l_max": expected_l_max,
        "l_cache": expected_l_cache,
    }


def test_context_budget_allows_overlong_fraction_overrides(tmp_path):
    config = tmp_path / "overlong.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "context_budget": {
                    "request_window_tokens": 32768,
                    "max_new_tokens_per_turn": 4096,
                    "max_turns": 30,
                    "generated_budget_fraction": 0.375,
                    "overlong_cache_fraction": 0.25,
                }
            }
        )
    )

    parsed = parse_rl_config(str(config))

    assert parsed.generator["trajectory_reward_shaping"]["overlong"] == {"l_max": 12288, "l_cache": 3072}


def test_resolved_context_budget_artifact_is_reproducible(tmp_path):
    parsed = parse_rl_config(str(_REPO_ROOT / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"))
    artifact = write_resolved_context_budget(
        parsed.context_budget, tmp_path / "resolved-context-budget.json", parsed.config_path
    )

    assert json.loads(artifact.read_text()) == {
        "config_path": str(parsed.config_path),
        "context_budget": {
            "generated_budget_fraction": 0.5,
            "generated_tokens_per_trajectory": 65536,
            "max_input_tokens": 114688,
            "max_new_tokens_per_turn": 16384,
            "max_turns": 90,
            "opencode_limit_context": 97280,
            "opencode_limit_output": 16384,
            "overlong_cache_fraction": 0.25,
            "overlong_cache_tokens": 16384,
            "request_window_tokens": 131072,
        },
    }

    remote_artifact = write_resolved_context_budget(
        parsed.context_budget,
        "memory://context-budget/resolved-context-budget.json",
        parsed.config_path,
    )
    assert remote_artifact == "memory://context-budget/resolved-context-budget.json"
    with fsspec.open(remote_artifact) as artifact_file:
        assert json.load(artifact_file)["context_budget"]["request_window_tokens"] == 131072


@pytest.mark.parametrize(
    ("window", "output", "expected_input", "expected_context"),
    [(131072, 16384, 114688, 97280), (32768, 4096, 28672, 23552)],
)
def test_opencode_limit_context_mirrors_harbor_formula(window, output, expected_input, expected_context):
    """Mirror harbor's _resolve_model_limit: context = input - output - min(1024, slack)."""
    budget = ContextBudget(request_window_tokens=window, max_new_tokens_per_turn=output, max_turns=30)

    assert budget.max_input_tokens == expected_input
    assert budget.opencode_limit_output == output
    assert budget.opencode_limit_context == expected_context
    assert budget.opencode_limit_context + budget.opencode_limit_output < budget.max_input_tokens
