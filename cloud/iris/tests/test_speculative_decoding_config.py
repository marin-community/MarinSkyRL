"""Behavior tests for the managed speculative-decoding configuration."""

from pathlib import Path
import sys
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
for source_root in (_REPO_ROOT, _REPO_ROOT / "skyrl-train"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from cloud.iris.rl_config_translation import build_skyrl_hydra_args, parse_rl_config  # noqa: E402
from marinskyrl.speculative_decoding import (  # noqa: E402
    SpeculativeDecodingConfigError,
    parse_speculative_decoding_config,
)
from skyrl_train.entrypoints.main_base import config_dir  # noqa: E402


_DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"


def _base_config() -> dict:
    return {
        "entrypoint": "standard",
        "context_budget": {
            "request_window_tokens": 2048,
            "max_new_tokens_per_turn": 512,
            "max_turns": 1,
        },
        "trainer": {"placement": {"colocate_all": False}},
        "generator": {
            "backend": "vllm",
            "run_engines_locally": True,
            "inference_engine_tensor_parallel_size": 1,
            "speculative_decoding": {
                "method": "eagle3",
                "model": {
                    "source_uri": "hf://laion/snowball-64k-eagle3-draft-r2egym",
                    "source_identity": _DRAFT_REVISION,
                },
                "num_speculative_tokens": 3,
                "training": {
                    "interval_steps": 1,
                    "reserved_gpu_memory_gib": 8,
                },
            },
        },
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "speculative.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_managed_speculator_reaches_hydra_with_immutable_source_unchanged(tmp_path: Path) -> None:
    parsed = parse_rl_config(str(_write_config(tmp_path, _base_config())))
    speculator = parsed.generator["speculative_decoding"]

    assert speculator["model"]["source_uri"] == "hf://laion/snowball-64k-eagle3-draft-r2egym"
    assert speculator["model"]["source_identity"] == _DRAFT_REVISION

    hydra_args = build_skyrl_hydra_args(parsed, {"num_nodes": 2}, SimpleNamespace(gpus_per_node=8))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=hydra_args)

    resolved = parse_speculative_decoding_config(
        OmegaConf.to_container(cfg.generator.speculative_decoding, resolve=True),
        backend=cfg.generator.backend,
        run_engines_locally=cfg.generator.run_engines_locally,
        entrypoint="skyrl_train.entrypoints.main_base",
        colocate_all=cfg.trainer.placement.colocate_all,
    )
    assert resolved is not None
    assert resolved.vllm_speculative_config() == {
        "method": "eagle3",
        "model": "laion/snowball-64k-eagle3-draft-r2egym",
        "num_speculative_tokens": 3,
        "revision": _DRAFT_REVISION,
    }
    assert resolved.training is not None
    assert resolved.training.interval_steps == 1
    assert resolved.training.reserved_gpu_memory_gib == 8


def test_materialized_speculator_uses_node_local_model_without_hub_resolution() -> None:
    value = _base_config()["generator"]["speculative_decoding"]
    value["model"]["materialized_path"] = "/tmp/draft-model"

    resolved = parse_speculative_decoding_config(
        value,
        backend="vllm",
        run_engines_locally=True,
        entrypoint="skyrl_train.entrypoints.main_base",
        colocate_all=False,
    )

    assert resolved is not None
    assert resolved.vllm_speculative_config() == {
        "method": "eagle3",
        "model": "/tmp/draft-model",
        "num_speculative_tokens": 3,
    }


def test_offline_speculator_training_is_supported_by_generate_entrypoint(tmp_path: Path) -> None:
    config = _base_config()
    config["entrypoint"] = "generate"

    parsed = parse_rl_config(str(_write_config(tmp_path, config)))

    assert parsed.entrypoint == "skyrl_train.entrypoints.main_generate"
    assert parsed.generator["speculative_decoding"]["training"] is not None


def test_null_speculator_keeps_the_default_disabled() -> None:
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")

    assert cfg.generator.speculative_decoding is None
    assert (
        parse_speculative_decoding_config(
            None,
            backend=cfg.generator.backend,
            run_engines_locally=cfg.generator.run_engines_locally,
            entrypoint="skyrl_train.entrypoints.main_base",
            colocate_all=cfg.trainer.placement.colocate_all,
        )
        is None
    )


def test_frozen_object_store_speculator_is_supported(tmp_path: Path) -> None:
    config = _base_config()
    model = config["generator"]["speculative_decoding"]["model"]
    model["source_uri"] = "s3://models/snowball/eagle3"
    model["source_identity"] = "snowball-eagle3@step-1888"
    config["generator"]["speculative_decoding"]["training"] = None

    parsed = parse_rl_config(str(_write_config(tmp_path, config)))

    assert parsed.generator["speculative_decoding"]["training"] is None
    assert parsed.generator["speculative_decoding"]["model"] == model

    resolved = parse_speculative_decoding_config(
        parsed.generator["speculative_decoding"],
        backend="vllm",
        run_engines_locally=True,
        entrypoint="skyrl_train.entrypoints.main_base",
        colocate_all=False,
    )
    assert resolved is not None
    assert resolved.vllm_speculative_config() == {
        "method": "eagle3",
        "model": "s3://models/snowball/eagle3",
        "num_speculative_tokens": 3,
        "draft_load_config": {"load_format": "runai_streamer"},
    }


def test_gcs_alias_is_normalized_for_vllm_runai_loading(tmp_path: Path) -> None:
    config = _base_config()
    model = config["generator"]["speculative_decoding"]["model"]
    model["source_uri"] = "gcs://models/snowball/eagle3"
    model["source_identity"] = "snowball-eagle3@step-1888"

    parsed = parse_rl_config(str(_write_config(tmp_path, config)))
    resolved = parse_speculative_decoding_config(
        parsed.generator["speculative_decoding"],
        backend="vllm",
        run_engines_locally=True,
        entrypoint="skyrl_train.entrypoints.main_base",
        colocate_all=False,
    )

    assert resolved is not None
    assert resolved.vllm_speculative_config()["model"] == "gs://models/snowball/eagle3"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_uri", "laion/draft", "must use hf://"),
        ("source_uri", "hf://laion", "must have the form"),
        ("source_identity", "main", "full 40-character"),
    ],
)
def test_speculator_rejects_nonreplayable_model_sources(tmp_path: Path, field: str, value: str, message: str) -> None:
    config = _base_config()
    config["generator"]["speculative_decoding"]["model"][field] = value

    with pytest.raises(SpeculativeDecodingConfigError, match=message):
        parse_rl_config(str(_write_config(tmp_path, config)))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("generator", "backend", "sglang"), "requires generator.backend=vllm"),
        (("generator", "run_engines_locally", False), "requires generator.run_engines_locally=true"),
        (("trainer", "placement", {"colocate_all": True}), "requires trainer.placement.colocate_all=false"),
        (("entrypoint", None, "fully_async"), "training is not supported"),
        (("entrypoint", None, "mini_swe"), "training is not supported"),
        (("entrypoint", None, "terminal_bench"), "training is not supported"),
    ],
)
def test_online_speculator_rejects_unsupported_execution_modes(
    tmp_path: Path, mutation: tuple[str, str | None, object], message: str
) -> None:
    config = _base_config()
    section, key, value = mutation
    if key is None:
        config[section] = value
    else:
        config[section][key] = value

    with pytest.raises(SpeculativeDecodingConfigError, match=message):
        parse_rl_config(str(_write_config(tmp_path, config)))


def test_online_speculator_requires_explicit_single_rank_inference(tmp_path: Path) -> None:
    config = _base_config()
    del config["generator"]["inference_engine_tensor_parallel_size"]

    with pytest.raises(SpeculativeDecodingConfigError, match="tensor_parallel_size=1"):
        parse_rl_config(str(_write_config(tmp_path, config)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_inference_engines": 2}, "num_inference_engines=1"),
        ({"tensor_parallel_size": 2}, "tensor_parallel_size=1"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size=1"),
        ({"async_engine": False}, "async_engine=true"),
        ({"engine_init_kwargs": {"async_scheduling": True}}, "async_scheduling"),
    ],
)
def test_online_speculator_rejects_unsupported_vllm_geometry(kwargs: dict, message: str) -> None:
    with pytest.raises(SpeculativeDecodingConfigError, match=message):
        parse_speculative_decoding_config(
            _base_config()["generator"]["speculative_decoding"],
            backend="vllm",
            run_engines_locally=True,
            entrypoint="skyrl_train.entrypoints.main_base",
            colocate_all=False,
            **kwargs,
        )


def test_raw_vllm_speculative_config_is_reserved(tmp_path: Path) -> None:
    config = _base_config()
    config["generator"].pop("speculative_decoding")
    config["generator"]["engine_init_kwargs"] = {"speculative_config": {"model": "/tmp/draft"}}

    with pytest.raises(ValueError, match="speculative_config"):
        parse_rl_config(str(_write_config(tmp_path, config)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("interval_steps", 0),
        ("max_window_tokens", 0),
        ("max_tokens_per_micro_batch", 0),
        ("holdout_fraction", 1),
        ("learning_rate", float("inf")),
        ("max_validation_loss_increase", -0.01),
        ("max_validation_agreement_decrease", float("nan")),
        ("reserved_gpu_memory_gib", 0),
    ],
)
def test_online_speculator_rejects_unbounded_or_unsupported_training_values(
    tmp_path: Path, field: str, value: object
) -> None:
    config = _base_config()
    config["generator"]["speculative_decoding"]["training"][field] = value

    with pytest.raises(SpeculativeDecodingConfigError, match=field):
        parse_rl_config(str(_write_config(tmp_path, config)))


def test_speculator_rejects_unknown_public_fields(tmp_path: Path) -> None:
    config = _base_config()
    config["generator"]["speculative_decoding"]["training"]["publish_interval_steps"] = 10

    with pytest.raises(SpeculativeDecodingConfigError, match="publish_interval_steps"):
        parse_rl_config(str(_write_config(tmp_path, config)))
