import pytest

from skyrl_train.model_wrapper import validate_grug_training_options
from skyrl_train.models.grug_moe import (
    GRUG_MOE_MODEL_TYPE,
    validate_grug_expert_parallel_options,
    validate_grug_selective_checkpoint_options,
    validate_grug_training_strategy,
)


_SUPPORTED_OPTIONS = {
    "model_type": GRUG_MOE_MODEL_TYPE,
    "attn_implementation": "eager",
    "use_sample_packing": False,
    "lora_rank": 0,
    "load_in_4bit": False,
    "sequence_parallel_size": 1,
    "context_parallel_size": 1,
    "moe_router_replay": False,
    "moe_grouped_gemm": False,
    "use_liger_kernel": False,
}


@pytest.mark.parametrize(
    ("option", "value", "label"),
    [
        ("attn_implementation", "sdpa", "attention backend"),
        ("use_sample_packing", True, "sample packing"),
        ("lora_rank", 8, "LoRA"),
        ("load_in_4bit", True, "4-bit loading"),
        ("sequence_parallel_size", 2, "sequence parallelism"),
        ("context_parallel_size", 2, "context parallelism"),
        ("moe_router_replay", True, "router replay/R3"),
        ("moe_grouped_gemm", True, "grouped MoE"),
        ("use_liger_kernel", True, "Liger kernels"),
    ],
)
def test_grug_training_options_reject_each_unsupported_feature(option, value, label):
    options = {**_SUPPORTED_OPTIONS, option: value}

    with pytest.raises(ValueError, match=label):
        validate_grug_training_options(**options)


@pytest.mark.parametrize("attn_implementation", ["eager", "flash_attention_2"])
def test_grug_training_options_accept_supported_attention(attn_implementation):
    validate_grug_training_options(**{**_SUPPORTED_OPTIONS, "attn_implementation": attn_implementation})


def test_grug_training_options_ignore_other_models():
    validate_grug_training_options(
        **{
            **_SUPPORTED_OPTIONS,
            "model_type": "qwen3",
            "attn_implementation": "flash_attention_2",
            "use_sample_packing": True,
        }
    )


@pytest.mark.parametrize("strategy", [None, "fsdp", "deepspeed", "megatron"])
def test_grug_training_strategy_rejects_non_fsdp2_backends(strategy):
    with pytest.raises(ValueError, match="trainer.strategy=fsdp2"):
        validate_grug_training_strategy(GRUG_MOE_MODEL_TYPE, strategy)


def test_grug_training_strategy_accepts_fsdp2_and_ignores_other_models():
    validate_grug_training_strategy(GRUG_MOE_MODEL_TYPE, "fsdp2")
    validate_grug_training_strategy("qwen3", "deepspeed")


def test_grug_expert_parallel_requires_native_grouped_torch_path():
    with pytest.raises(ValueError, match="use_grouped_mm=true"):
        validate_grug_expert_parallel_options(
            GRUG_MOE_MODEL_TYPE,
            expert_model_parallel_size=2,
            use_grouped_mm=False,
            ep_comm_backend="torch",
        )
    with pytest.raises(ValueError, match="ep_comm_backend=torch"):
        validate_grug_expert_parallel_options(
            GRUG_MOE_MODEL_TYPE,
            expert_model_parallel_size=2,
            use_grouped_mm=True,
            ep_comm_backend="deepep",
        )

    validate_grug_expert_parallel_options(
        GRUG_MOE_MODEL_TYPE,
        expert_model_parallel_size=2,
        use_grouped_mm=True,
        ep_comm_backend="torch",
    )
    validate_grug_expert_parallel_options(
        GRUG_MOE_MODEL_TYPE,
        expert_model_parallel_size=1,
        use_grouped_mm=False,
        ep_comm_backend="deepep",
    )


def test_grug_selective_checkpoint_requires_supported_execution():
    options = {
        "model_type": GRUG_MOE_MODEL_TYPE,
        "enabled": True,
        "gradient_checkpointing": True,
        "use_reentrant": False,
        "use_grouped_mm": True,
    }
    validate_grug_selective_checkpoint_options(**options)

    invalid = (
        ({"model_type": "qwen3"}, "Grug policy model"),
        ({"gradient_checkpointing": False}, "gradient_checkpointing=true"),
        ({"use_reentrant": True}, "gradient_checkpointing_use_reentrant=false"),
        ({"use_grouped_mm": False}, "use_grouped_mm=true"),
    )
    for override, message in invalid:
        with pytest.raises(ValueError, match=message):
            validate_grug_selective_checkpoint_options(**{**options, **override})

    validate_grug_selective_checkpoint_options(**{**options, "enabled": False, "model_type": "qwen3"})
