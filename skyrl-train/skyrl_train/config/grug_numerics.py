"""The Grug policy's forward numerics, ``trainer.policy.megatron_config.grug_numerics``."""

from enum import StrEnum

from omegaconf import DictConfig

GRUG_NUMERICS_KEY = "trainer.policy.megatron_config.grug_numerics"


class GrugNumerics(StrEnum):
    """Megatron's own kernels, or the bytes a decode-invariant vLLM engine computes for every token."""

    MEGATRON = "megatron"
    VLLM_DECODE_INVARIANT = "vllm_decode_invariant"


def _one_layer_recompute_units(transformer_config_kwargs: DictConfig) -> bool:
    return transformer_config_kwargs.get("recompute_granularity") == "full" and (
        transformer_config_kwargs.get("recompute_method") == "block"
        or transformer_config_kwargs.get("recompute_num_layers") == 1
    )


def validate_grug_numerics_config(cfg: DictConfig) -> GrugNumerics:
    """The configured numerics, which ``VLLM_DECODE_INVARIANT`` accepts only in the setup it was verified in.

    The trainer computes what a decode-invariant engine (``generator.decode_invariant``) computes at temperature 1, on
    unpacked sequences, with one GPU per tensor- and context-parallel group and per expert tensor-parallel group,
    Transformer Engine's fused attention for the gradient, and activation recompute either off or in one-layer units.
    """
    numerics = GrugNumerics(cfg.trainer.policy.megatron_config.grug_numerics)
    if numerics is GrugNumerics.MEGATRON:
        return numerics
    megatron = cfg.trainer.policy.megatron_config
    problems = []
    if not cfg.generator.decode_invariant:
        problems.append("generator.decode_invariant=true")
    if cfg.generator.sampling_params.temperature != 1.0:
        problems.append("generator.sampling_params.temperature=1")
    if cfg.trainer.use_sample_packing:
        problems.append("trainer.use_sample_packing=false")
    if cfg.trainer.flash_attn:
        problems.append("trainer.flash_attn=false")
    if megatron.tensor_model_parallel_size != 1 or megatron.context_parallel_size != 1:
        problems.append("tensor_model_parallel_size=1 and context_parallel_size=1")
    if megatron.get("expert_tensor_parallel_size") not in (None, 1):
        problems.append("expert_tensor_parallel_size=1")
    if cfg.trainer.gradient_checkpointing and not _one_layer_recompute_units(megatron.transformer_config_kwargs):
        problems.append(
            "full activation recompute in one-layer units (recompute_method=block, or recompute_num_layers=1) or "
            "trainer.gradient_checkpointing=false"
        )
    if problems:
        raise ValueError(f"{GRUG_NUMERICS_KEY}={numerics} needs {'; '.join(problems)}")
    return numerics
