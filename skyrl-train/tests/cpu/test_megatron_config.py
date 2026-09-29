import json
import sys
from unittest import mock

from omegaconf import OmegaConf
import pytest
from ci.pivot_grug_smoke import MODEL, MODEL_REVISION, compare_arms, launch_config, main, validate_smoke_config
from cloud.iris.launch_config import load_launch_config
from infra.rl_data.pivot_swe import DATASET_ID, DATASET_REVISION
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_cfg


def _megatron_replay_cfg():
    cfg = get_default_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    return cfg


def test_megatron_router_replay_reaches_dcp_guard():
    """Enabled Megatron replay counts as active R3 capture at the DCP guard."""
    cfg = _megatron_replay_cfg()
    cfg.generator.inference_engine_decode_context_parallel_size = 2
    cfg.generator.inference_engine_tensor_parallel_size = 8

    with (
        mock.patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("offline")),
        pytest.raises(AssertionError, match="decode context parallel.*R3 router capture"),
    ):
        validate_cfg(cfg)


def test_unsupported_strategy_rejected():
    cfg = get_default_config()
    cfg.trainer.strategy = "unknown"
    cfg.trainer.logger = "console"

    with pytest.raises(ValueError, match="Unsupported training strategy"):
        validate_cfg(cfg)


def test_megatron_router_replay_rejects_fused_router():
    cfg = _megatron_replay_cfg()
    # Struct mode blocks new dict keys; the launcher's Hydra `+` override allows
    # exactly this addition, so mirror it here.
    OmegaConf.set_struct(cfg, False)
    cfg.trainer.policy.megatron_config.transformer_config_kwargs.moe_router_fusion = True

    with pytest.raises(ValueError, match="moe_router_fusion"):
        validate_cfg(cfg)


@pytest.mark.parametrize("role", ["policy", "ref"])
def test_grug_smoke_rejects_context_parallel_sliding_window_before_launch(tmp_path, role):
    config = launch_config(
        "grug-cp-regression",
        str(tmp_path / "output"),
        str(tmp_path / "temporary"),
        "s3://marin-us-east-02a/preflight/grug",
        MODEL_REVISION,
    )
    # Packing passed the generic CP check but then failed inside TE attention on the GPUs.
    config.skyrl.trainer.use_sample_packing = True
    config.skyrl.trainer[role].megatron_config.context_parallel_size = 2
    path = tmp_path / "launch.yaml"
    OmegaConf.save(config, path)
    resolved = load_launch_config(path)

    with pytest.raises(ValueError, match=rf"trainer\.{role}\.megatron_config\.context_parallel_size=1"):
        validate_smoke_config(resolved)


@pytest.mark.parametrize("role", ["policy", "ref"])
def test_grug_smoke_rejects_uneven_pipeline_without_stage_layout(tmp_path, role):
    config = launch_config(
        "grug-pipeline-regression",
        str(tmp_path / "output"),
        str(tmp_path / "temporary"),
        "s3://marin-us-east-02a/preflight/grug",
        MODEL_REVISION,
    )
    # Grug has 26 layers, so PP4 needs an explicit uneven stage partition.
    del config.skyrl.trainer[role].megatron_config.transformer_config_kwargs.num_layers_in_first_pipeline_stage
    del config.skyrl.trainer[role].megatron_config.transformer_config_kwargs.num_layers_in_last_pipeline_stage
    path = tmp_path / "launch.yaml"
    OmegaConf.save(config, path)

    with pytest.raises(ValueError, match="26 layers cannot be divided"):
        validate_smoke_config(load_launch_config(path))


@pytest.mark.parametrize("role", ["policy", "ref"])
def test_grug_smoke_rejects_tp5_vocab_partition_before_launch(tmp_path, role):
    config = launch_config(
        "grug-vocab-regression",
        str(tmp_path / "output"),
        str(tmp_path / "temporary"),
        "s3://marin-us-east-02a/preflight/grug",
        MODEL_REVISION,
        recipe_path="cloud/iris/configs/grug_pivot_swe_64gpu.yaml",
    )
    config.skyrl.trainer[role].megatron_config.tensor_model_parallel_size = 5
    path = tmp_path / "launch.yaml"
    OmegaConf.save(config, path)

    with pytest.raises(ValueError, match="128256 vocabulary tokens cannot be divided"):
        validate_smoke_config(load_launch_config(path))


def test_grug_retry_preflight_preserves_sample_model_and_token_budget(tmp_path, monkeypatch, capsys):
    source_root = tmp_path / "source"
    data_root = source_root / "data"
    source_config = launch_config(
        "source-sft",
        str(source_root / "sft"),
        str(tmp_path / "temporary-source"),
        "s3://marin-us-east-02a/cached-model",
        "sha256:pinned-model",
        arm="sft",
        data_root=str(data_root),
        recipe_path="cloud/iris/configs/grug_pivot_swe_smoke.yaml",
    )
    source_config.skyrl.trainer.pivot_token_budget = 1_421_216_000
    (source_root / "sft").mkdir(parents=True)
    (source_root / "sft" / "resolved-launch.yaml").write_text(
        json.dumps({"config": OmegaConf.to_container(source_config, resolve=True)})
    )
    (source_root / "diagnostics").mkdir()
    (source_root / "diagnostics" / "budget.json").write_text(
        json.dumps(
            {
                "dataset": DATASET_ID,
                "revision": DATASET_REVISION,
                "tokenizer": MODEL,
                "tokenizer_revision": MODEL_REVISION,
                "learner_token_budget": 1_421_216_000,
                "train_prefixes": 512,
                "nominal_updates": 10,
                "max_prompt_tokens": 32256,
                "max_reference_tokens": 512,
                "sequences_per_full_update": 8192,
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pivot_grug_smoke",
            "--run-id",
            "retry-rl",
            "--output-root",
            str(tmp_path / "retry-output"),
            "--temporary-root",
            str(tmp_path / "retry-temporary"),
            "--recipe",
            "cloud/iris/configs/grug_pivot_swe_64gpu.yaml",
            "--cluster",
            "cw-rno2a",
            "--reuse-comparison-root",
            str(source_root),
        ],
    )

    main()

    retry = OmegaConf.create(capsys.readouterr().out)
    assert retry.inputs.model.uri == source_config.inputs.model.uri
    assert retry.inputs.model.identity == source_config.inputs.model.identity
    assert retry.inputs.train_data[0].uri == str(data_root)
    assert retry.inputs.validation_data[0].uri == str(data_root)
    assert retry.skyrl.trainer.pivot_token_budget == 1_421_216_000
    assert retry.iris.cluster == "cw-rno2a"
    assert retry.iris.allocation.num_nodes == 8
    assert retry.iris.allocation.gpus_per_node == 8


def test_grug_comparison_pairs_outputs_from_separate_runs(tmp_path):
    arm_outputs = {arm: str(tmp_path / arm) for arm in ("sft", "rl")}
    for arm, scores in {"rl": (1.0, 0.0), "sft": (0.0, 1.0)}.items():
        diagnostics = tmp_path / arm / "diagnostics"
        diagnostics.mkdir(parents=True)
        rows = [
            {
                "trajectory_id": trajectory_id,
                "before": {"score": [0.0], "exception_type": None},
                "after": {"score": [score], "exception_type": None},
            }
            for trajectory_id, score in zip((11, 22), scores)
        ]
        (diagnostics / "comparison.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "learner_token_budget": 1000,
        "revision": DATASET_REVISION,
        "train_prefixes": 2,
        "probe_trajectory_ids": [11, 22],
    }
    arm_summaries = {
        arm: {
            "training_input_tokens": 1000,
            "training_response_tokens": 100,
            "training_responses_per_step": {1: 2},
        }
        for arm in arm_outputs
    }
    comparison_root = tmp_path / "paired"
    (comparison_root / "diagnostics").mkdir(parents=True)

    summary = compare_arms(str(comparison_root), arm_outputs, manifest, arm_summaries)

    assert summary["rl_wins"] == 1
    assert summary["sft_wins"] == 1
    assert summary["probe_count"] == 2
    assert summary["training_input_tokens"] == {"rl": 1000, "sft": 1000}
    paired = [
        json.loads(line) for line in (comparison_root / "diagnostics" / "comparison.jsonl").read_text().splitlines()
    ]
    assert paired == [
        {"trajectory_id": 11, "rl_before": 0.0, "rl_after": 1.0, "sft_before": 0.0, "sft_after": 0.0},
        {"trajectory_id": 22, "rl_before": 0.0, "rl_after": 0.0, "sft_before": 0.0, "sft_after": 1.0},
    ]
