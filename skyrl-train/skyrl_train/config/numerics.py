"""How the trainer and the rollout engines compute, ``trainer.numerics``."""

from dataclasses import dataclass
from enum import StrEnum

from omegaconf import DictConfig, OmegaConf
from transformers import PretrainedConfig

from skyrl_train.config.weight_sync_pause import resolve_weight_sync_pause_policy
from skyrl_train.inference_engines.vllm.decode_invariant.engine import verify_decode_invariant_config

NUMERICS_KEY = "trainer.numerics"
CHECK_KEY = f"{NUMERICS_KEY}.check"
# Megatron's token dispatcher the exact trainer's expert-parallel combine runs on.
ALLTOALL_DISPATCHER = "alltoall"
# Megatron's recompute granularity that checkpoints whole layers.
FULL_RECOMPUTE = "full"
# The deepest pipeline the exact trainer's stage statistic hand-off is verified on.
MAX_PIPELINE_STAGES = 2
# Megatron settings the exact trainer takes from the model config, as the engines take theirs: a
# transformer_config_kwargs override would change the trainer's model alone.
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


class NumericsMode(StrEnum):
    """The kernels the trainer and the rollout engines run: each side's own (``native``), vLLM's batch-invariant
    kernels on both sides (``batch_invariant``), or decode-invariant vLLM engines whose log-probabilities the
    trainer reproduces bit for bit (``exact``, ``inference_engines/vllm/decode_invariant`` and
    ``models/grug_exact``)."""

    NATIVE = "native"
    BATCH_INVARIANT = "batch_invariant"
    EXACT = "exact"

    @property
    def bit_identical(self) -> bool:
        """True for a mode that promises the trainer the engines' log-probabilities bit for bit."""
        return self is NumericsMode.EXACT


class CheckFailure(StrEnum):
    """What the exactness check does when a token's log-probability differs: log it and continue, or stop the run."""

    LOG = "log"
    STOP = "stop"


@dataclass(frozen=True)
class ExactnessCheckConfig:
    """When the online exactness check runs (``every_weight_syncs`` 0 = never) and what a mismatch does."""

    every_weight_syncs: int
    on_failure: CheckFailure

    @property
    def enabled(self) -> bool:
        return self.every_weight_syncs > 0


@dataclass(frozen=True)
class NumericsResolution:
    """The numerics mode a run uses, and why a configured ``exact`` resolved to ``native`` (empty otherwise)."""

    mode: NumericsMode
    fallback_reasons: tuple[str, ...] = ()


def read_policy_model_config(cfg: DictConfig) -> PretrainedConfig:
    """The policy's model config, read from the config file at ``trainer.policy.model.path`` without running model
    code."""
    # skyrl_train.models imports skyrl_train.utils, whose validate_cfg imports this module.
    from skyrl_train.models.grug_moe import GRUG_MOE_MODEL_TYPE, GrugMoeConfig  # noqa: PLC0415

    model = cfg.trainer.policy.model
    values, _ = PretrainedConfig.get_config_dict(model.path, revision=model.get("revision"))
    config_class = GrugMoeConfig if values.get("model_type") == GRUG_MOE_MODEL_TYPE else PretrainedConfig
    return config_class.from_dict(values)


def recomputes_one_layer_per_unit(granularity: str | None, method: str | None, num_layers: int | None) -> bool:
    """True when Megatron's full activation recompute checkpoints every layer as a unit of its own."""
    return granularity == FULL_RECOMPUTE and method == "uniform" and num_layers == 1


def exact_model_problems(model_config: PretrainedConfig) -> list[str]:
    """Why the policy model is not one the exact trainer's kernels compute: Grug with the compiled shapes and norm
    epsilon, and no Hero layers, ShortConv or latent experts."""
    # Imported here for the same cycle as in read_policy_model_config.
    from skyrl_train.models.grug_exact.inductor_kernels import COMPILED_MODEL  # noqa: PLC0415
    from skyrl_train.models.grug_moe import GRUG_MOE_MODEL_TYPE  # noqa: PLC0415

    if model_config.model_type != GRUG_MOE_MODEL_TYPE:
        return [f"the policy is {model_config.model_type}, not Grug"]
    problems = [
        f"the policy's {name} is {getattr(model_config, name)}, not the compiled kernels' {value}"
        for name, value in COMPILED_MODEL.items()
        if getattr(model_config, name) != value
    ]
    if model_config.uses_hero_architecture:
        problems.append("the policy uses Hero layers, ShortConv or latent experts")
    return problems


def resolve_exact_numerics(policy_model_config: PretrainedConfig) -> NumericsResolution:
    """What a configured ``exact`` resolves to for a policy with ``policy_model_config``: ``exact``, or ``native``
    with every reason ``exact_model_problems`` finds."""
    reasons = exact_model_problems(policy_model_config)
    if reasons:
        return NumericsResolution(NumericsMode.NATIVE, tuple(reasons))
    return NumericsResolution(NumericsMode.EXACT)


def read_numerics(cfg: DictConfig) -> NumericsResolution:
    """The numerics a run of ``cfg`` uses: its configured mode, with a configured ``exact`` resolved against the
    policy's model config file (``resolve_exact_numerics``). Never writes to ``cfg``."""
    mode = NumericsMode(cfg.trainer.numerics.mode)
    if mode is not NumericsMode.EXACT:
        return NumericsResolution(mode)
    return resolve_exact_numerics(read_policy_model_config(cfg))


def exactness_check_config(cfg: DictConfig) -> ExactnessCheckConfig:
    """The exactness check settings of ``cfg``."""
    check = cfg.trainer.numerics.check
    return ExactnessCheckConfig(int(check.every_weight_syncs), CheckFailure(check.on_failure))


def validate_numerics_config(cfg: DictConfig) -> None:
    """Accept a known mode, ``batch_invariant`` and ``exact`` only with local vLLM engines, ``exact`` only with
    settings a decode-invariant engine and the exact trainer run on and in-memory rollout hand-off, and check
    settings the mode supports."""
    numerics = cfg.trainer.numerics
    try:
        NumericsMode(numerics.mode)
    except ValueError as error:
        choices = ", ".join(mode.value for mode in NumericsMode)
        raise ValueError(f"{NUMERICS_KEY}.mode must be one of {choices}; got {numerics.mode!r}") from error
    resolution = read_numerics(cfg)
    mode = resolution.mode
    if mode is not NumericsMode.NATIVE:
        if cfg.generator.backend != "vllm":
            raise ValueError(f"{NUMERICS_KEY}.mode={mode} requires generator.backend='vllm'")
        if not cfg.generator.run_engines_locally:
            raise ValueError(
                f"{NUMERICS_KEY}.mode={mode} cannot configure a remote inference server; run the vLLM "
                "engines locally so both rollout and trainer activation is guaranteed"
            )
    if mode is NumericsMode.EXACT:
        _validate_exact_numerics(cfg)
        _validate_exact_trainer(cfg)
    if int(numerics.check.every_weight_syncs) < 0:
        raise ValueError(f"{CHECK_KEY}.every_weight_syncs must be at least 0; got {numerics.check.every_weight_syncs}")
    try:
        check = exactness_check_config(cfg)
    except ValueError as error:
        choices = ", ".join(failure.value for failure in CheckFailure)
        raise ValueError(
            f"{CHECK_KEY}.on_failure must be one of {choices}; got {numerics.check.on_failure!r}"
        ) from error
    if check.enabled and cfg.generator.backend != "vllm":
        raise ValueError(f"the exactness check requires generator.backend='vllm'; got {cfg.generator.backend!r}")
    if check.on_failure is CheckFailure.STOP and resolution.fallback_reasons:
        raise ValueError(
            f"{CHECK_KEY}.on_failure=stop requires bit-identical log-probabilities, and the configured "
            f"{NUMERICS_KEY}.mode=exact resolves to {mode} for this policy: " + "; ".join(resolution.fallback_reasons)
        )
    if check.on_failure is CheckFailure.STOP and not mode.bit_identical:
        raise ValueError(
            f"{CHECK_KEY}.on_failure=stop requires a numerics mode that promises bit-identical log-probabilities; "
            f"{NUMERICS_KEY}.mode={mode} does not"
        )


def _validate_exact_numerics(cfg: DictConfig) -> None:
    """Raise unless the rollout engines of ``cfg`` can be decode-invariant and hand their rollouts to the trainer in
    memory: the trainer must receive the engines' log-probabilities and routes in memory, and the object-store path
    is not covered by the exactness tests."""
    if cfg.trainer.rollout_buffer.object_store_root is not None:
        raise ValueError(f"{NUMERICS_KEY}.mode=exact requires trainer.rollout_buffer.object_store_root to be unset")
    if cfg.generator.get("speculative_decoding") is not None:
        raise ValueError(f"{NUMERICS_KEY}.mode=exact runs without speculative decoding")
    try:
        verify_decode_invariant_config(
            backend=cfg.generator.backend,
            attention_backend=cfg.generator.get("vllm_attention_backend"),
            enforce_eager=cfg.generator.enforce_eager,
            tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
            decode_context_parallel_size=cfg.generator.inference_engine_decode_context_parallel_size,
            engine_init_kwargs=OmegaConf.to_container(cfg.generator.engine_init_kwargs, resolve=True),
        )
    except ValueError as error:
        raise ValueError(f"{NUMERICS_KEY}.mode=exact: {error}") from error


def exact_trainer_problems(cfg: DictConfig) -> list[str]:
    """The settings of ``cfg`` the exact trainer does not run on."""
    generator = cfg.generator
    trainer = cfg.trainer
    problems = []
    if generator.get("rope_scaling") or generator.get("rope_theta") is not None:
        problems.append(
            "generator.rope_scaling and generator.rope_theta must be unset: the rotary table is the model config's"
        )
    if generator.sampling_params.temperature != 1.0:
        problems.append("generator.sampling_params.temperature must be 1")
    if not resolve_weight_sync_pause_policy(generator).clear_cache:
        problems.append(
            "generator.weight_sync_pause.clear_cache must be true: a kept prefix cache holds old weights' values"
        )
    if trainer.use_sample_packing:
        problems.append("trainer.use_sample_packing must be false")
    if trainer.flash_attn:
        problems.append("trainer.flash_attn must be false")
    megatron = trainer.policy.megatron_config
    for name in ("tensor_model_parallel_size", "context_parallel_size"):
        if megatron[name] != 1:
            problems.append(f"{name} must be 1")
    if megatron.get("expert_tensor_parallel_size") not in (None, 1):
        problems.append("expert_tensor_parallel_size must be 1")
    if megatron.pipeline_model_parallel_size > MAX_PIPELINE_STAGES:
        problems.append(f"pipeline_model_parallel_size must be at most {MAX_PIPELINE_STAGES}")
    if megatron.ddp_config.overlap_param_gather:
        problems.append(
            "ddp_config.overlap_param_gather must be false: the exact kernels read norm and expert weights outside "
            "the modules whose forward waits for the parameter all-gather"
        )
    transformer = megatron.transformer_config_kwargs
    if transformer.get("moe_token_dispatcher_type", ALLTOALL_DISPATCHER) != ALLTOALL_DISPATCHER:
        problems.append(f"moe_token_dispatcher_type must be {ALLTOALL_DISPATCHER}")
    problems.extend(
        f"transformer_config_kwargs.{name} must be unset: the trainer takes it from the model config, as the engines do"
        for name in MODEL_CONFIG_TRANSFORMER_SETTINGS
        if name in transformer
    )
    if trainer.gradient_checkpointing and not recomputes_one_layer_per_unit(
        transformer.get("recompute_granularity"),
        transformer.get("recompute_method"),
        transformer.get("recompute_num_layers"),
    ):
        problems.append(
            "activation recompute must be off or checkpoint one layer per unit (recompute_granularity=full, "
            "recompute_method=uniform, recompute_num_layers=1)"
        )
    return problems


def _validate_exact_trainer(cfg: DictConfig) -> None:
    problems = exact_trainer_problems(cfg)
    if problems:
        raise ValueError(f"{NUMERICS_KEY}.mode=exact: " + "; ".join(problems))
