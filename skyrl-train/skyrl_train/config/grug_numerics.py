"""The Grug policy's forward numerics, ``trainer.policy.megatron_config.grug_numerics``."""

from enum import StrEnum

from omegaconf import DictConfig

GRUG_NUMERICS_KEY = "trainer.policy.megatron_config.grug_numerics"
# The engine's routed experts run vLLM's Triton fused-MoE kernels, which the trainer computes each slot with.
TRITON_MOE_BACKEND = "triton"
# vLLM's expert-parallel combine whose order the trainer reproduces: an all-gather, then a reduce-scatter.
ALLGATHER_REDUCESCATTER = "allgather_reducescatter"


class GrugNumerics(StrEnum):
    """Megatron's own kernels, or the bytes a decode-invariant vLLM engine computes for every token."""

    MEGATRON = "megatron"
    VLLM_DECODE_INVARIANT = "vllm_decode_invariant"


def one_layer_recompute_units(granularity: str | None, method: str | None, num_layers: int | None) -> bool:
    """True when Megatron's full activation recompute checkpoints every layer as a unit of its own: ``recompute_method``
    ``block``, or ``uniform`` with one layer per unit."""
    return granularity == "full" and (method == "block" or num_layers == 1)


def validate_grug_numerics_config(cfg: DictConfig) -> GrugNumerics:
    """The configured numerics, which ``VLLM_DECODE_INVARIANT`` accepts only in the setup it was verified in.

    The trainer computes what a decode-invariant engine (``generator.decode_invariant``) computes at temperature 1 with
    vLLM's Triton MoE kernels and all-gather/reduce-scatter expert parallelism, on unpacked sequences, with one GPU per
    tensor- and context-parallel group and per expert tensor-parallel group, Transformer Engine's fused attention for the
    gradient, and activation recompute either off or in one-layer units.
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
    engine = cfg.generator.engine_init_kwargs
    if engine.get("moe_backend") != TRITON_MOE_BACKEND:
        problems.append(f"generator.engine_init_kwargs.moe_backend={TRITON_MOE_BACKEND}")
    if engine.get("all2all_backend", ALLGATHER_REDUCESCATTER) != ALLGATHER_REDUCESCATTER:
        problems.append(f"generator.engine_init_kwargs.all2all_backend={ALLGATHER_REDUCESCATTER}")
    if cfg.trainer.use_sample_packing:
        problems.append("trainer.use_sample_packing=false")
    if cfg.trainer.flash_attn:
        problems.append("trainer.flash_attn=false")
    if megatron.tensor_model_parallel_size != 1 or megatron.context_parallel_size != 1:
        problems.append("tensor_model_parallel_size=1 and context_parallel_size=1")
    if megatron.get("expert_tensor_parallel_size") not in (None, 1):
        problems.append("expert_tensor_parallel_size=1")
    recompute = megatron.transformer_config_kwargs
    if cfg.trainer.gradient_checkpointing and not one_layer_recompute_units(
        recompute.get("recompute_granularity"), recompute.get("recompute_method"), recompute.get("recompute_num_layers")
    ):
        problems.append(
            "full activation recompute in one-layer units (recompute_method=block, or recompute_num_layers=1) or "
            "trainer.gradient_checkpointing=false"
        )
    if problems:
        raise ValueError(f"{GRUG_NUMERICS_KEY}={numerics} needs {'; '.join(problems)}")
    return numerics
