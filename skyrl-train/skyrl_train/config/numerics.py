"""How the policy's trainer and its rollout engines compute, ``trainer.algorithm.numerics``."""

from dataclasses import dataclass
from enum import StrEnum

from omegaconf import DictConfig

from skyrl_train.config.decode_invariant import decode_invariant_engine_problems
from skyrl_train.config.weight_sync_pause import resolve_weight_sync_pause_policy

NUMERICS_KEY = "trainer.algorithm.numerics"
# The engine's routed experts run vLLM's Triton fused-MoE kernels, which the trainer computes each slot with.
TRITON_MOE_BACKEND = "triton"
# vLLM's expert-parallel combine whose order the trainer reproduces: an all-gather, then a reduce-scatter.
ALLGATHER_REDUCESCATTER = "allgather_reducescatter"


class Numerics(StrEnum):
    """Each side's own kernels (``native``), vLLM's batch-invariant kernels on both sides (``batch_invariant``), or
    decode-invariant vLLM engines and a Grug trainer that computes their bytes for every token (``exact``)."""

    NATIVE = "native"
    BATCH_INVARIANT = "batch_invariant"
    EXACT = "exact"


class ExactnessFailure(StrEnum):
    """What a failed exactness check does under ``exact`` numerics: stop the run, or log the failure and continue."""

    STOP = "stop"
    LOG = "log"


@dataclass(frozen=True)
class NumericsResolution:
    """The numerics a run uses, and the configured numerics it fell back from."""

    numerics: Numerics
    fallback_from: Numerics | None = None


def resolve_numerics(cfg: DictConfig) -> NumericsResolution:
    """The configured numerics; ``exact`` falls back to ``native`` when weight syncs keep the engines' prefix caches
    (``generator.weight_sync_pause.clear_cache=false``), whose entries hold the previous weights' bytes."""
    numerics = Numerics(cfg.trainer.algorithm.numerics)
    if numerics is Numerics.EXACT and not resolve_weight_sync_pause_policy(cfg.generator).clear_cache:
        return NumericsResolution(Numerics.NATIVE, fallback_from=Numerics.EXACT)
    return NumericsResolution(numerics)


def one_layer_recompute_units(granularity: str | None, method: str | None, num_layers: int | None) -> bool:
    """True when Megatron's full activation recompute checkpoints every layer as a unit of its own: ``recompute_method``
    ``block``, or ``uniform`` with one layer per unit."""
    return granularity == "full" and (method == "block" or num_layers == 1)


def _exact_problems(cfg: DictConfig) -> list[str]:
    generator = cfg.generator
    problems = decode_invariant_engine_problems(
        backend=generator.backend,
        attention_backend=generator.get("vllm_attention_backend"),
        enforce_eager=generator.enforce_eager,
        tensor_parallel_size=generator.inference_engine_tensor_parallel_size,
        decode_context_parallel_size=generator.get("inference_engine_decode_context_parallel_size", 1),
    )
    if not generator.run_engines_locally:
        problems.append("generator.run_engines_locally=true")
    engine = generator.engine_init_kwargs
    if engine.get("moe_backend") != TRITON_MOE_BACKEND:
        problems.append(f"generator.engine_init_kwargs.moe_backend={TRITON_MOE_BACKEND}")
    if engine.get("all2all_backend", ALLGATHER_REDUCESCATTER) != ALLGATHER_REDUCESCATTER:
        problems.append(f"generator.engine_init_kwargs.all2all_backend={ALLGATHER_REDUCESCATTER}")
    if generator.sampling_params.temperature != 1.0:
        problems.append("generator.sampling_params.temperature=1")
    trainer = cfg.trainer
    if trainer.strategy != "megatron":
        problems.append("trainer.strategy=megatron")
    if trainer.use_sample_packing:
        problems.append("trainer.use_sample_packing=false")
    if trainer.flash_attn:
        problems.append("trainer.flash_attn=false")
    megatron = trainer.policy.megatron_config
    if megatron.tensor_model_parallel_size != 1 or megatron.context_parallel_size != 1:
        problems.append("tensor_model_parallel_size=1 and context_parallel_size=1")
    if megatron.get("expert_tensor_parallel_size") not in (None, 1):
        problems.append("expert_tensor_parallel_size=1")
    recompute = megatron.transformer_config_kwargs
    if trainer.gradient_checkpointing and not one_layer_recompute_units(
        recompute.get("recompute_granularity"), recompute.get("recompute_method"), recompute.get("recompute_num_layers")
    ):
        problems.append(
            "full activation recompute in one-layer units (recompute_method=block, or recompute_num_layers=1) or "
            "trainer.gradient_checkpointing=false"
        )
    check = trainer.algorithm.exactness_check
    if int(check.every_weight_syncs) < 0:
        problems.append("trainer.algorithm.exactness_check.every_weight_syncs >= 0")
    if check.on_failure not in set(ExactnessFailure):
        problems.append(f"trainer.algorithm.exactness_check.on_failure in {sorted(ExactnessFailure)}")
    return problems


def validate_numerics_config(cfg: DictConfig) -> None:
    """Accept ``batch_invariant`` and ``exact`` numerics only in the setups they were verified in.

    ``exact`` needs decode-invariant engines (compiled local vLLM engines with FLASH_ATTN at TP 1, Triton MoE kernels
    and all-gather/reduce-scatter expert parallelism) sampling at temperature 1, and a Megatron trainer on unpacked
    sequences with one GPU per tensor-, context- and expert tensor-parallel group, Transformer Engine's fused attention
    for the gradient, and activation recompute off or in one-layer units.
    """
    numerics = Numerics(cfg.trainer.algorithm.numerics)
    if numerics is Numerics.BATCH_INVARIANT:
        if cfg.generator.backend != "vllm":
            raise ValueError(f"{NUMERICS_KEY}=batch_invariant requires generator.backend='vllm'")
        if not cfg.generator.run_engines_locally:
            raise ValueError(
                f"{NUMERICS_KEY}=batch_invariant cannot configure a remote inference server; run the vLLM engines "
                "locally so both rollout and trainer activation is guaranteed"
            )
    if numerics is Numerics.EXACT and (problems := _exact_problems(cfg)):
        raise ValueError(f"{NUMERICS_KEY}=exact needs {'; '.join(problems)}")
