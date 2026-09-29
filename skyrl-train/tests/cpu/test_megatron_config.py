import json
import sys
from unittest import mock

from omegaconf import OmegaConf
import pytest
from ci.pivot_grug_smoke import MODEL, MODEL_REVISION, launch_config, main, validate_smoke_config
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


def test_grug_retry_preflight_preserves_sample_model_and_token_budget(tmp_path, monkeypatch, capsys):
    recipe = "cloud/iris/configs/grug_pivot_swe_tp5_retry.yaml"
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
        recipe_path=recipe,
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
            recipe,
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
    assert retry.iris.allocation.num_nodes == 6
