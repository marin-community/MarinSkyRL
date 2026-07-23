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
    apply_context_budget_overrides,
    build_skyrl_hydra_args,
    parse_rl_config,
    write_resolved_context_budget,
)


@dataclass
class _HPCStub:
    gpus_per_node: int = 8


_CONFIGS = {
    "128GPU_80B_A3B_next_cp1.yaml": (98304, 16384, 999999),
    "32GPU_qwen3_coder_30b_a3b_ep4.yaml": (131072, 16384, 999999),
    "32GPU_qwen3_coder_30b_a3b_ep4_nooffload.yaml": (131072, 16384, 999999),
    "56GPU_qwen3_8b.yaml": (32768, 4096, 999999),
    "64GPU_qwen3_6_35b_a3b.yaml": (131072, 16384, 999999),
    "delphi_math_rl.yaml": (4096, 3584, 1),
    "delphi_math_rl_ifeval.yaml": (4096, 3584, 1),
    "opencode_smoke_literal.yaml": (32768, 4096, 30),
    "tasktrove_dq_sweep_30b.yaml": (32768, 4096, 30),
    "tasktrove_dq_sweep_30b_ncclnet.yaml": (32768, 4096, 30),
    "tasktrove_dq_sweep_30b_terminus2.yaml": (32768, 4096, 30),
}


def test_all_iris_configs_materialize_one_coherent_context_budget():
    configs_dir = _REPO_ROOT / "cloud/iris/configs"
    assert {path.name for path in configs_dir.glob("*.yaml")} == set(_CONFIGS)

    for name, (window, output, turns) in _CONFIGS.items():
        parsed = parse_rl_config(str(configs_dir / name))
        assert parsed.context_budget.request_window_tokens == window
        assert parsed.context_budget.max_new_tokens_per_turn == output
        assert parsed.context_budget.max_turns == turns
        assert (
            parsed.trainer["max_prompt_length"] + parsed.generator["sampling_params"]["max_generate_length"] == window
        )
        assert parsed.generator["max_input_length"] == parsed.context_budget.max_input_tokens
        assert parsed.generator["engine_init_kwargs"]["max_model_len"] == window
        assert parsed.generator["max_turns"] == turns
        if parsed.terminal_bench is not None:
            assert parsed.terminal_bench["harbor"]["max_turns"] == turns
            assert parsed.terminal_bench["model_info"] == {
                "max_input_tokens": parsed.context_budget.max_input_tokens,
                "max_output_tokens": output,
            }


def test_context_budget_derives_all_hydra_length_arguments():
    parsed = parse_rl_config(str(_REPO_ROOT / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"))
    args = build_skyrl_hydra_args(parsed, {"job_name": "context-test", "num_nodes": 4}, _HPCStub())

    assert "trainer.max_prompt_length=28672" in args
    assert "generator.max_input_length=28672" in args
    assert "generator.max_turns=30" in args
    assert "generator.sampling_params.max_generate_length=4096" in args
    assert any(arg.endswith("generator.engine_init_kwargs.max_model_len=32768") for arg in args)
    assert "+terminal_bench_config.model_info.max_input_tokens=28672" in args
    assert "+terminal_bench_config.model_info.max_output_tokens=4096" in args
    assert "+terminal_bench_config.harbor.max_turns=30" in args


def test_context_budget_rejects_impossible_and_legacy_config_fields(tmp_path):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        yaml.safe_dump(
            {
                "context_budget": {
                    "request_window_tokens": 4096,
                    "max_new_tokens_per_turn": 4096,
                    "max_turns": 1,
                }
            }
        )
    )
    with pytest.raises(ValueError, match="must exceed"):
        parse_rl_config(str(invalid))

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        yaml.safe_dump(
            {
                "context_budget": {
                    "request_window_tokens": 32768,
                    "max_new_tokens_per_turn": 4096,
                    "max_turns": 30,
                },
                "generator": {"max_input_length": 32000},
            }
        )
    )
    with pytest.raises(ValueError, match="generator.max_input_length"):
        parse_rl_config(str(legacy))


def test_context_budget_override_rederives_lengths_and_rejects_low_level_fields():
    parsed = parse_rl_config(str(_REPO_ROOT / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"))
    overridden, passthrough = apply_context_budget_overrides(
        parsed,
        ["context_budget.max_new_tokens_per_turn=2048", "trainer.logger=console"],
    )

    assert overridden.context_budget.max_input_tokens == 30720
    assert overridden.generator["sampling_params"]["max_generate_length"] == 2048
    assert overridden.terminal_bench["model_info"]["max_output_tokens"] == 2048
    assert passthrough == ["trainer.logger=console"]

    with pytest.raises(ValueError, match="derived from context_budget"):
        apply_context_budget_overrides(parsed, ["generator.engine_init_kwargs.max_model_len=65536"])


def test_resolved_context_budget_artifact_is_reproducible(tmp_path):
    parsed = parse_rl_config(str(_REPO_ROOT / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"))
    artifact = write_resolved_context_budget(
        parsed.context_budget, tmp_path / "resolved-context-budget.json", parsed.config_path
    )

    assert json.loads(artifact.read_text()) == {
        "config_path": str(parsed.config_path),
        "context_budget": {
            "max_input_tokens": 28672,
            "max_new_tokens_per_turn": 4096,
            "max_turns": 30,
            "request_window_tokens": 32768,
        },
    }

    remote_artifact = write_resolved_context_budget(
        parsed.context_budget,
        "memory://context-budget/resolved-context-budget.json",
        parsed.config_path,
    )
    assert remote_artifact == "memory://context-budget/resolved-context-budget.json"
    with fsspec.open(remote_artifact) as artifact_file:
        assert json.load(artifact_file)["context_budget"]["request_window_tokens"] == 32768
