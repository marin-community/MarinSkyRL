"""How the policy's trainer and its rollout engines compute, ``trainer.algorithm.numerics``.

Config resolution (``apply_numerics_resolution``) keeps ``exact`` only for a setup that the decode-invariant engines and
the Grug trainer numerics support, and otherwise resolves the run to ``native`` and records the reasons in
``trainer.algorithm.numerics_fallback_reasons``. Engines and trainer read ``trainer.algorithm.resolved_numerics``, which
only config resolution sets.
"""

from dataclasses import dataclass
from enum import StrEnum

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, PretrainedConfig

from skyrl_train.config.behavior_logprobs import behavior_logprob_problems
from skyrl_train.config.decode_invariant import (
    DECODE_INVARIANT_ATTENTION_BACKEND,
    decode_invariant_engine_problems,
    flash_attn_version_override,
)
from skyrl_train.config.grug_vllm_shapes import (
    HEAD_DIM,
    HEADS,
    HIDDEN,
    KV_HEADS,
    QUERY_FACTORS,
    RMS_NORM_EPS,
    ROTARY_POSITIONS,
    SHARED_WIDTH,
    VOCAB,
)
from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.config.weight_sync_pause import resolve_weight_sync_pause_policy

NUMERICS_KEY = "trainer.algorithm.numerics"
# The engine's routed experts run vLLM's Triton fused-MoE kernels, which the trainer computes each slot with.
TRITON_MOE_BACKEND = "triton"
# vLLM's expert-parallel combine whose order the trainer reproduces.
ALLGATHER_REDUCESCATTER = "allgather_reducescatter"
# Megatron's token dispatcher the trainer's expert-parallel numerics run on.
ALLTOALL_DISPATCHER = "alltoall"
# The deepest pipeline the trainer's stage statistic hand-off supports.
MAX_PIPELINE_STAGES = 2
# The Grug model config attributes that compiled vLLM's kernels hard-code, and their Snowball values.
COMPILED_MODEL_VALUES = {
    "hidden_size": HIDDEN,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "shared_expert_intermediate_size": SHARED_WIDTH,
    "qk_mult": QUERY_FACTORS[0],
    "qk_mult_long_scale": QUERY_FACTORS[1],
    "rms_norm_eps": RMS_NORM_EPS,
}
# Megatron settings the trainer takes from the model config, as the engines take theirs: a transformer_config_kwargs
# override changes the trainer's model alone.
MODEL_CONFIG_TRANSFORMER_SETTINGS = (
    "hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "kv_channels",
    "layernorm_epsilon",
    "softmax_scale",
    "window_size",
    "window_attn_skip_freq",
    "no_rope_freq",
    "rotary_base",
    "rotary_interleaved",
    "ffn_hidden_size",
    "moe_ffn_hidden_size",
    "moe_shared_expert_intermediate_size",
    "num_moe_experts",
    "moe_router_topk",
    "grug_qk_mult",
    "grug_qk_mult_long_scale",
)


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
    """The numerics a run uses, and why ``exact`` fell back to ``native`` (empty when it did not)."""

    numerics: Numerics
    fallback_reasons: tuple[str, ...] = ()


def one_layer_recompute_units(granularity: str | None, method: str | None, num_layers: int | None) -> bool:
    """True when Megatron's full activation recompute checkpoints every layer as a unit of its own: ``recompute_method``
    ``block``, or ``uniform`` with one layer per unit."""
    return granularity == "full" and (method == "block" or num_layers == 1)


def exact_setup_problems(cfg: DictConfig, runner_mode: TrajectoryRunnerMode) -> list[str]:
    """Why the settings of ``cfg`` and the trajectory runner are not a setup ``exact`` numerics support.

    An unset attention backend or MoE backend is accepted, since ``configure_exact_engines`` sets it.
    """
    generator = cfg.generator
    engine = generator.engine_init_kwargs
    problems = [
        f"needs {setting}"
        for setting in decode_invariant_engine_problems(
            backend=generator.backend,
            attention_backend=generator.get("vllm_attention_backend") or DECODE_INVARIANT_ATTENTION_BACKEND,
            flash_attn_version=flash_attn_version_override(engine),
            enforce_eager=generator.enforce_eager,
            tensor_parallel_size=generator.inference_engine_tensor_parallel_size,
            decode_context_parallel_size=generator.get("inference_engine_decode_context_parallel_size", 1),
        )
    ]
    if not generator.run_engines_locally:
        problems.append("needs generator.run_engines_locally=true")
    if generator.get("rope_scaling") or generator.get("rope_theta") is not None:
        problems.append(
            "needs no generator.rope_scaling or generator.rope_theta: the trainer's rotary table is the model config's"
        )
    longest = generator.max_input_length + generator.sampling_params.max_generate_length
    if longest > ROTARY_POSITIONS:
        problems.append(
            f"needs generator.max_input_length + generator.sampling_params.max_generate_length <= {ROTARY_POSITIONS}: "
            f"the RoPE kernel reads positions below it, and sequences can reach {longest}"
        )
    if engine.get("moe_backend", TRITON_MOE_BACKEND) != TRITON_MOE_BACKEND:
        problems.append(f"needs generator.engine_init_kwargs.moe_backend={TRITON_MOE_BACKEND}")
    if engine.get("all2all_backend", ALLGATHER_REDUCESCATTER) != ALLGATHER_REDUCESCATTER:
        problems.append(f"needs generator.engine_init_kwargs.all2all_backend={ALLGATHER_REDUCESCATTER}")
    if "compilation_config" in engine:
        problems.append(
            "needs no generator.engine_init_kwargs.compilation_config: decode-invariant engines set their own"
        )
    if generator.get("speculative_decoding") is not None:
        problems.append("needs generator.speculative_decoding=null")
    if generator.sampling_params.temperature != 1.0:
        problems.append("needs generator.sampling_params.temperature=1")
    problems.extend(f"sampling: {problem}" for problem in behavior_logprob_problems(generator))
    if not resolve_weight_sync_pause_policy(generator).clear_cache:
        problems.append(
            "needs generator.weight_sync_pause.clear_cache=true: a kept prefix cache holds old weights' bytes"
        )
    if runner_mode is not TrajectoryRunnerMode.SKYRL_GYM:
        problems.append(f"the {runner_mode} trajectory runner does not report each sequence's serving engine rank")
    trainer = cfg.trainer
    if trainer.use_sample_packing:
        problems.append("needs trainer.use_sample_packing=false")
    if trainer.flash_attn:
        problems.append("needs trainer.flash_attn=false")
    megatron = trainer.policy.megatron_config
    for name in ("tensor_model_parallel_size", "context_parallel_size"):
        if megatron[name] != 1:
            problems.append(f"needs {name}=1")
    if megatron.get("expert_tensor_parallel_size") not in (None, 1):
        problems.append("needs expert_tensor_parallel_size=1")
    if megatron.pipeline_model_parallel_size > MAX_PIPELINE_STAGES:
        problems.append(f"needs pipeline_model_parallel_size<={MAX_PIPELINE_STAGES}")
    if megatron.ddp_config.overlap_param_gather:
        problems.append(
            "needs ddp_config.overlap_param_gather=false: the trainer's kernels read norm and expert weights outside "
            "the modules whose forward waits for the parameter all-gather"
        )
    transformer = megatron.transformer_config_kwargs
    if transformer.get("moe_token_dispatcher_type", ALLTOALL_DISPATCHER) != ALLTOALL_DISPATCHER:
        problems.append(f"needs moe_token_dispatcher_type={ALLTOALL_DISPATCHER}")
    problems.extend(
        f"needs no transformer_config_kwargs.{name}: the trainer takes it from the model config, as the engines do"
        for name in MODEL_CONFIG_TRANSFORMER_SETTINGS
        if name in transformer
    )
    if trainer.gradient_checkpointing and not one_layer_recompute_units(
        transformer.get("recompute_granularity"),
        transformer.get("recompute_method"),
        transformer.get("recompute_num_layers"),
    ):
        problems.append(
            "needs full activation recompute in one-layer units (recompute_method=block, or recompute_num_layers=1) "
            "or trainer.gradient_checkpointing=false"
        )
    return problems


def policy_model_config(model_path: str, revision: str | None) -> PretrainedConfig:
    """The policy's model config, read from local files only."""
    # skyrl_train.models imports skyrl_train.utils, whose validate_cfg imports this module.
    from skyrl_train.models import register_local_models  # noqa: PLC0415

    register_local_models()
    return AutoConfig.from_pretrained(model_path, revision=revision, trust_remote_code=True, local_files_only=True)


def exact_model_problems(model_config: PretrainedConfig) -> list[str]:
    """Why the policy model is not one that compiled vLLM's kernels compute: Grug with Snowball's shapes and norm
    epsilon, a vocabulary within Snowball's, and no Hero layers, ShortConv or latent experts."""
    # skyrl_train.models imports skyrl_train.utils, whose validate_cfg imports this module.
    from skyrl_train.models import GRUG_MOE_MODEL_TYPE  # noqa: PLC0415

    if model_config.model_type != GRUG_MOE_MODEL_TYPE:
        return [f"the policy is {model_config.model_type}, not Grug"]
    problems = [
        f"the policy's {name} is {getattr(model_config, name)}, not Snowball's {value}"
        for name, value in COMPILED_MODEL_VALUES.items()
        if getattr(model_config, name) != value
    ]
    if model_config.vocab_size > VOCAB:
        problems.append(
            f"the policy's vocab_size is {model_config.vocab_size}, above Snowball's {VOCAB}, below which the "
            "embedding kernel reads token ids"
        )
    if model_config.uses_hero_architecture:
        problems.append("the policy uses Hero layers, ShortConv or latent experts")
    return problems


def resolve_numerics(cfg: DictConfig, runner_mode: TrajectoryRunnerMode) -> NumericsResolution:
    """The numerics a run of ``cfg`` with ``runner_mode`` uses. ``exact`` falls back to ``native`` for every reason
    ``exact_setup_problems`` finds; when there is none, for every reason ``exact_model_problems`` finds in the policy's
    model config, or when that config cannot be read."""
    numerics = Numerics(cfg.trainer.algorithm.numerics)
    if numerics is not Numerics.EXACT:
        return NumericsResolution(numerics)
    reasons = exact_setup_problems(cfg, runner_mode)
    if not reasons:
        model = cfg.trainer.policy.model
        try:
            model_config = policy_model_config(model.path, model.get("revision"))
        except (OSError, ValueError) as error:
            reasons = [f"the policy's model config could not be read: {error}"]
        else:
            reasons = exact_model_problems(model_config)
    if reasons:
        return NumericsResolution(Numerics.NATIVE, tuple(reasons))
    return NumericsResolution(Numerics.EXACT)


def configure_exact_engines(generator: DictConfig) -> None:
    """Set the engine settings ``exact`` needs that ``generator`` leaves unset: the attention and MoE backends."""
    if generator.get("vllm_attention_backend") is None:
        OmegaConf.update(generator, "vllm_attention_backend", DECODE_INVARIANT_ATTENTION_BACKEND, force_add=True)
    if "moe_backend" not in generator.engine_init_kwargs:
        OmegaConf.update(generator.engine_init_kwargs, "moe_backend", TRITON_MOE_BACKEND, force_add=True)


def apply_numerics_resolution(cfg: DictConfig, runner_mode: TrajectoryRunnerMode) -> NumericsResolution:
    """Resolve the run's numerics into ``trainer.algorithm.resolved_numerics`` and ``numerics_fallback_reasons``, set
    the engine settings a resolved ``exact`` needs, and warn about a fallback from ``exact``."""
    resolution = resolve_numerics(cfg, runner_mode)
    algorithm = cfg.trainer.algorithm
    algorithm.resolved_numerics = str(resolution.numerics)
    algorithm.numerics_fallback_reasons = list(resolution.fallback_reasons)
    if resolution.numerics is Numerics.EXACT:
        configure_exact_engines(cfg.generator)
    if resolution.fallback_reasons:
        logger.warning(
            f"{NUMERICS_KEY}=exact falls back to native, so trainer and rollout log-probabilities will differ: "
            + "; ".join(resolution.fallback_reasons)
        )
    return resolution


def validate_numerics_config(cfg: DictConfig) -> None:
    """Accept ``batch_invariant`` numerics only with local vLLM engines, and only valid exactness check settings."""
    if Numerics(cfg.trainer.algorithm.resolved_numerics) is Numerics.BATCH_INVARIANT:
        if cfg.generator.backend != "vllm":
            raise ValueError(f"{NUMERICS_KEY}=batch_invariant requires generator.backend='vllm'")
        if not cfg.generator.run_engines_locally:
            raise ValueError(
                f"{NUMERICS_KEY}=batch_invariant cannot configure a remote inference server; run the vLLM engines "
                "locally so both rollout and trainer activation is guaranteed"
            )
    check = cfg.trainer.algorithm.exactness_check
    if int(check.every_weight_syncs) < 0:
        raise ValueError("trainer.algorithm.exactness_check.every_weight_syncs must be at least 0")
    ExactnessFailure(check.on_failure)
