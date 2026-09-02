"""RL training configuration parsing + Hydra-argument translation for MarinSkyRL.

Provides YAML-based configuration for SkyRL RL training jobs, replacing 50+ Hydra
CLI arguments with a single ``--rl_config`` YAML file.

Usage::

    from cloud.iris.rl_config_translation import parse_rl_config, build_skyrl_hydra_args

    parsed = parse_rl_config("configs/56gpu_qwen3_8b.yaml")
    hydra_args = build_skyrl_hydra_args(parsed, exp_args, hpc)
"""

from __future__ import annotations

import base64
import binascii
import copy
import fsspec
import json
import math
import os
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol

import yaml

from cloud.iris.paths import resolve_paths_in_dict
from marinskyrl.resource_locator import join_resource_path, model_source_for_path

# Directory containing the bundled example RL config YAML files.
SKYRL_CONFIG_DIR = Path(__file__).parent / "configs"
RL_CONFIG_TASK_DIR = "/tmp/marin-rl-configs"
RL_CONFIG_PAYLOAD_ENV = "MARIN_RL_CONFIG_B64"


class RLEntrypoint(StrEnum):
    """Execution modes supported by Iris RL configurations."""

    FULLY_ASYNC = "fully_async"
    GENERATE = "generate"
    MINI_SWE = "mini_swe"
    STANDARD = "standard"
    TERMINAL_BENCH = "terminal_bench"
    TERMINAL_BENCH_GENERATE = "terminal_bench_generate"
    TERMINAL_BENCH_TEACHER_LOGITS = "terminal_bench_teacher_logits"


RL_ENTRYPOINT_MODULES = {
    RLEntrypoint.FULLY_ASYNC: "skyrl_train.entrypoints.fully_async",
    RLEntrypoint.GENERATE: "skyrl_train.entrypoints.main_generate",
    RLEntrypoint.MINI_SWE: "skyrl_train.entrypoints.mini_swe",
    RLEntrypoint.STANDARD: "skyrl_train.entrypoints.main_base",
    RLEntrypoint.TERMINAL_BENCH: "skyrl_train.entrypoints.terminal_bench",
    RLEntrypoint.TERMINAL_BENCH_GENERATE: "skyrl_train.entrypoints.terminal_bench_generate",
    RLEntrypoint.TERMINAL_BENCH_TEACHER_LOGITS: "skyrl_train.entrypoints.terminal_bench_teacher_logits",
}


def resolve_rl_entrypoint(value: str | None, *, config_path: Path) -> str:
    """Resolve one supported RL execution mode to its packaged module."""
    name = RLEntrypoint.STANDARD if value is None else value
    try:
        entrypoint = RLEntrypoint(name)
    except ValueError as error:
        choices = ", ".join(item.value for item in RLEntrypoint)
        raise ValueError(
            f"{config_path}: entrypoint must be a registered name ({choices}); got {name!r}. "
            "Python module paths are not accepted in RL configs."
        ) from error

    return RL_ENTRYPOINT_MODULES[entrypoint]


class HPCGeometry(Protocol):
    """Hardware geometry required while translating a launch configuration."""

    gpus_per_node: int


_REQUIRED_CONTEXT_BUDGET_FIELDS = frozenset(
    {
        "request_window_tokens",
        "max_new_tokens_per_turn",
        "max_turns",
    }
)
_CONTEXT_BUDGET_FRACTION_FIELDS = frozenset({"generated_budget_fraction", "overlong_cache_fraction"})
_CONTEXT_BUDGET_FIELDS = _REQUIRED_CONTEXT_BUDGET_FIELDS | _CONTEXT_BUDGET_FRACTION_FIELDS
_DEFAULT_GENERATED_BUDGET_FRACTION = 0.5
_DEFAULT_OVERLONG_CACHE_FRACTION = 0.25

_DERIVED_CONTEXT_FIELDS = (
    ("trainer", "max_prompt_length"),
    ("generator", "max_input_length"),
    ("generator", "max_turns"),
    ("generator", "sampling_params", "max_generate_length"),
    ("generator", "engine_init_kwargs", "max_model_len"),
    ("terminal_bench", "harbor", "max_episodes"),
    ("terminal_bench", "harbor", "max_turns"),
    ("terminal_bench", "harbor", "llm_call_kwargs", "max_tokens"),
    ("terminal_bench", "model_info", "max_input_tokens"),
    ("terminal_bench", "model_info", "max_output_tokens"),
    ("generator", "trajectory_reward_shaping", "overlong", "l_max"),
    ("generator", "trajectory_reward_shaping", "overlong", "l_cache"),
)


@dataclass(frozen=True)
class ContextBudget:
    """One coherent token budget for an Iris RL rollout request."""

    request_window_tokens: int
    max_new_tokens_per_turn: int
    max_turns: int
    generated_budget_fraction: float = _DEFAULT_GENERATED_BUDGET_FRACTION
    overlong_cache_fraction: float = _DEFAULT_OVERLONG_CACHE_FRACTION

    @property
    def max_input_tokens(self) -> int:
        """Return the input allowance after reserving one complete response."""
        return self.request_window_tokens - self.max_new_tokens_per_turn

    @property
    def opencode_limit_output(self) -> int:
        """OpenCode's per-request output cap (mirrors harbor ``_resolve_model_limit``)."""
        return min(self.max_new_tokens_per_turn, max(1, self.max_input_tokens - 1))

    @property
    def opencode_limit_context(self) -> int:
        """OpenCode's sliding-window / compaction-trigger size.

        Mirrors the formula in ``harbor/src/harbor/agents/installed/opencode.py``
        ``_resolve_model_limit``: ``context = window - output - margin`` where
        ``margin`` reserves a small safety band so ``context + output`` stays
        strictly below the engine's prompt cap.
        """
        output = self.opencode_limit_output
        margin = min(1024, max(0, self.max_input_tokens - output - 1))
        return max(1, self.max_input_tokens - output - margin)

    @property
    def generated_tokens_per_trajectory(self) -> int:
        """Return the generated-token allowance used by trajectory-level shaping."""
        if self.max_turns == 1:
            return self.max_new_tokens_per_turn
        return max(1, int(self.request_window_tokens * self.generated_budget_fraction))

    @property
    def overlong_cache_tokens(self) -> int:
        """Return the soft-overlong transition width."""
        return int(self.generated_tokens_per_trajectory * self.overlong_cache_fraction)

    def as_dict(self) -> Dict[str, int | float]:
        """Return the persisted representation, including derived client input."""
        return {
            "request_window_tokens": self.request_window_tokens,
            "max_new_tokens_per_turn": self.max_new_tokens_per_turn,
            "max_turns": self.max_turns,
            "generated_budget_fraction": self.generated_budget_fraction,
            "overlong_cache_fraction": self.overlong_cache_fraction,
            "max_input_tokens": self.max_input_tokens,
            "generated_tokens_per_trajectory": self.generated_tokens_per_trajectory,
            "overlong_cache_tokens": self.overlong_cache_tokens,
            "opencode_limit_context": self.opencode_limit_context,
            "opencode_limit_output": self.opencode_limit_output,
        }


def _path_is_declared(mapping: Dict[str, Any], path: tuple[str, ...]) -> bool:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def _validate_no_derived_context_fields(raw: Dict[str, Any], config_path: Path) -> None:
    declared = [".".join(path) for path in _DERIVED_CONTEXT_FIELDS if _path_is_declared(raw, path)]
    if declared:
        raise ValueError(
            f"{config_path} declares derived context fields: {', '.join(declared)}. "
            "Declare only context_budget instead."
        )


def _remove_derived_context_fields(raw: Dict[str, Any]) -> None:
    for path in _DERIVED_CONTEXT_FIELDS:
        parent: Any = raw
        for key in path[:-1]:
            if not isinstance(parent, dict) or key not in parent:
                parent = None
                break
            parent = parent[key]
        if isinstance(parent, dict):
            parent.pop(path[-1], None)


def _require_positive_integer(value: Any, field_name: str, config_path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{config_path}: context_budget.{field_name} must be a positive integer, got {value!r}")
    return value


def _require_fraction(value: Any, field_name: str, config_path: Path, *, allow_zero: bool) -> float:
    valid = False
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        lower_bound_satisfied = value >= 0 if allow_zero else value > 0
        valid = math.isfinite(value) and lower_bound_satisfied and value <= 1
    if not valid:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{config_path}: context_budget.{field_name} must be in {interval}, got {value!r}")
    return float(value)


def resolve_context_budget(raw: Dict[str, Any], config_path: Path) -> ContextBudget:
    """Validate and resolve the single public context budget declaration.

    The request window includes the prompt and the current response. The derived
    client input limit therefore reserves the complete per-turn output allowance.
    """
    _validate_no_derived_context_fields(raw, config_path)
    config = raw.get("context_budget")
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: context_budget must be a mapping")

    unknown = set(config) - _CONTEXT_BUDGET_FIELDS
    if unknown:
        raise ValueError(f"{config_path}: unknown context_budget fields: {', '.join(sorted(unknown))}")
    missing = _REQUIRED_CONTEXT_BUDGET_FIELDS - set(config)
    if missing:
        raise ValueError(f"{config_path}: missing context_budget fields: {', '.join(sorted(missing))}")

    budget = ContextBudget(
        request_window_tokens=_require_positive_integer(
            config["request_window_tokens"], "request_window_tokens", config_path
        ),
        max_new_tokens_per_turn=_require_positive_integer(
            config["max_new_tokens_per_turn"], "max_new_tokens_per_turn", config_path
        ),
        max_turns=_require_positive_integer(config["max_turns"], "max_turns", config_path),
        generated_budget_fraction=_require_fraction(
            config.get("generated_budget_fraction", _DEFAULT_GENERATED_BUDGET_FRACTION),
            "generated_budget_fraction",
            config_path,
            allow_zero=False,
        ),
        overlong_cache_fraction=_require_fraction(
            config.get("overlong_cache_fraction", _DEFAULT_OVERLONG_CACHE_FRACTION),
            "overlong_cache_fraction",
            config_path,
            allow_zero=True,
        ),
    )
    if budget.max_input_tokens <= 0:
        raise ValueError(
            f"{config_path}: request_window_tokens ({budget.request_window_tokens}) must exceed "
            f"max_new_tokens_per_turn ({budget.max_new_tokens_per_turn})"
        )
    return budget


def _materialize_context_budget(
    raw: Dict[str, Any], budget: ContextBudget
) -> tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Return SkyRL sections populated from one resolved context budget."""
    trainer = copy.deepcopy(raw.get("trainer", {}))
    generator = copy.deepcopy(raw.get("generator", {}))
    terminal_bench = copy.deepcopy(raw.get("terminal_bench"))
    materialized_raw = copy.deepcopy(raw)

    trainer["max_prompt_length"] = budget.max_input_tokens
    generator["max_input_length"] = budget.max_input_tokens
    generator["max_turns"] = budget.max_turns
    generator.setdefault("sampling_params", {})["max_generate_length"] = budget.max_new_tokens_per_turn
    generator.setdefault("engine_init_kwargs", {})["max_model_len"] = budget.request_window_tokens
    generator.setdefault("trajectory_reward_shaping", {})["overlong"] = {
        "l_max": budget.generated_tokens_per_trajectory,
        "l_cache": budget.overlong_cache_tokens,
    }

    if terminal_bench is not None:
        harbor = terminal_bench.setdefault("harbor", {})
        harbor["max_turns"] = budget.max_turns
        harbor.setdefault("llm_call_kwargs", {})["max_tokens"] = budget.max_new_tokens_per_turn
        model_info = terminal_bench.get("model_info") or {}
        model_info["max_input_tokens"] = budget.max_input_tokens
        model_info["max_output_tokens"] = budget.max_new_tokens_per_turn
        terminal_bench["model_info"] = model_info

    materialized_raw["context_budget"] = budget.as_dict()
    materialized_raw["trainer"] = copy.deepcopy(trainer)
    materialized_raw["generator"] = copy.deepcopy(generator)
    if terminal_bench is not None:
        materialized_raw["terminal_bench"] = copy.deepcopy(terminal_bench)
    return trainer, generator, terminal_bench, materialized_raw


def _override_key(override: str) -> str:
    key, separator, _value = override.lstrip("+").partition("=")
    if not separator:
        raise ValueError(f"Invalid SkyRL override {override!r}; expected KEY=VALUE")
    return key


def apply_context_budget_overrides(
    parsed: "ParsedRLConfig", overrides: List[str]
) -> tuple["ParsedRLConfig", List[str]]:
    """Resolve high-level context overrides and reject derived-field overrides.

    `--skyrl_override` is the existing user-facing launcher mechanism. Context
    values are consumed here instead of reaching Hydra, whose schema deliberately
    has no `context_budget` node.
    """
    values = parsed.context_budget.as_dict()
    passthrough: List[str] = []
    derived_names = {".".join(path) for path in _DERIVED_CONTEXT_FIELDS}

    for override in overrides:
        key = _override_key(override)
        if key.startswith("context_budget."):
            field_name = key.removeprefix("context_budget.")
            if field_name not in _CONTEXT_BUDGET_FIELDS:
                raise ValueError(f"Unsupported context budget override {key!r}")
            raw_value = override.partition("=")[2]
            try:
                if field_name in _CONTEXT_BUDGET_FRACTION_FIELDS:
                    values[field_name] = float(raw_value)
                else:
                    values[field_name] = int(raw_value)
            except ValueError as error:
                expected_type = "a number" if field_name in _CONTEXT_BUDGET_FRACTION_FIELDS else "an integer"
                raise ValueError(f"{key} must be {expected_type}, got {raw_value!r}") from error
            continue
        if key in derived_names:
            raise ValueError(
                f"{key} is derived from context_budget and cannot be overridden directly. "
                "Override context_budget.request_window_tokens, context_budget.max_new_tokens_per_turn, "
                "context_budget.max_turns, context_budget.generated_budget_fraction, or "
                "context_budget.overlong_cache_fraction instead."
            )
        passthrough.append(override)

    raw = copy.deepcopy(parsed.raw)
    raw["trainer"] = copy.deepcopy(parsed.trainer)
    raw["generator"] = copy.deepcopy(parsed.generator)
    if parsed.terminal_bench is not None:
        raw["terminal_bench"] = copy.deepcopy(parsed.terminal_bench)
    _remove_derived_context_fields(raw)
    raw["context_budget"] = {field: values[field] for field in _CONTEXT_BUDGET_FIELDS}
    budget = resolve_context_budget(raw, parsed.config_path)
    trainer, generator, terminal_bench, materialized_raw = _materialize_context_budget(raw, budget)
    return (
        replace(
            parsed,
            raw=materialized_raw,
            trainer=trainer,
            generator=generator,
            terminal_bench=terminal_bench,
            context_budget=budget,
            tensor_parallel_size=generator.get("inference_engine_tensor_parallel_size", 1),
        ),
        passthrough,
    )


def write_resolved_context_budget(budget: ContextBudget, destination: Path | str, config_path: Path) -> Path | str:
    """Persist the resolved context contract for an Iris RL launch."""
    payload = (
        json.dumps(
            {
                "config_path": str(config_path),
                "context_budget": budget.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if isinstance(destination, Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)
        return destination
    with fsspec.open(destination, "w") as artifact:
        artifact.write(payload)
    return destination


# =============================================================================
# SkyRL Internal Engine Kwargs - DO NOT SET IN YAML CONFIGS
# =============================================================================
# These kwargs are set internally by SkyRL and will cause "duplicate keyword
# argument" errors if also specified in engine_init_kwargs.
#
# Source: skyrl_train/inference_engines/ray_wrapped_inference_engine.py
# =============================================================================

SKYRL_INTERNAL_ENGINE_KWARGS = frozenset(
    {
        # Hardcoded values
        "trust_remote_code",  # Always True
        "worker_extension_cls",  # vLLM SkyRL extension path
        "data_parallel_backend",  # Hardcoded "mp"
        "max_logprobs",  # Hardcoded 1
        # Calculated from config/environment
        "distributed_executor_backend",  # Calculated from TP size ("uni" or "ray")
        "enforce_eager",  # Set from generator.enforce_eager config
        "tensor_parallel_size",  # Set from generator config
        "data_parallel_size",  # Set from generator config
        "seed",  # Set from config
        "enable_prefix_caching",  # Set from generator config
        "dtype",  # Set from generator.model_dtype
        "gpu_memory_utilization",  # Set from generator config
        "max_num_batched_tokens",  # Set from generator config
        "max_num_seqs",  # Set from generator config
        "enable_sleep_mode",  # Set from trainer.placement.colocate_all
        "vllm_v1_disable_multiproc",  # Set from generator config
        # Ray internal management
        "bundle_indices",  # Calculated from parallelism config
        "num_gpus",  # Ray resource allocation
        "noset_visible_devices",  # Ray CUDA_VISIBLE_DEVICES handling
        # SGLang-specific (if using SGLang backend)
        "model_path",  # Set from trainer.policy.model.path
        "tp_size",  # Alias for tensor_parallel_size
        "mem_fraction_static",  # Alias for gpu_memory_utilization
        "random_seed",  # Alias for seed
        "disable_radix_cache",  # Inverse of enable_prefix_caching
        "max_prefill_tokens",  # Alias for max_num_batched_tokens
        "max_running_requests",  # Alias for max_num_seqs
        "mm_attention_backend",  # Hardcoded "fa3"
        "attention_backend",  # Hardcoded "fa3"
        "enable_memory_saver",  # Set from inference_engine_enable_sleep
        "tokenizer",  # Passed from external tokenizer
        "custom_weight_loader",  # Hardcoded SkyRL path
        "skip_tokenizer_init",  # Hardcoded True for SGLang
    }
)


def validate_engine_init_kwargs(
    engine_init_kwargs: Dict[str, Any],
    config_path: Optional[Path] = None,
) -> None:
    """Fail fast if ``engine_init_kwargs`` contains SkyRL-internal keys.

    SkyRL sets certain vLLM/SGLang engine kwargs internally; specifying them in
    the YAML config causes "duplicate keyword argument" errors at runtime.

    Raises:
        ValueError: If any forbidden keys are found in engine_init_kwargs.
    """
    if not engine_init_kwargs:
        return

    forbidden_found = set(engine_init_kwargs.keys()) & SKYRL_INTERNAL_ENGINE_KWARGS

    if forbidden_found:
        config_context = f" in {config_path}" if config_path else ""
        forbidden_list = "\n".join(f"  - {k}" for k in sorted(forbidden_found))
        all_forbidden = "\n".join(f"  - {k}" for k in sorted(SKYRL_INTERNAL_ENGINE_KWARGS))

        raise ValueError(
            f"engine_init_kwargs{config_context} contains keys that SkyRL sets internally.\n"
            f"These will cause 'duplicate keyword argument' errors at runtime.\n\n"
            f"FORBIDDEN KEYS FOUND:\n{forbidden_list}\n\n"
            f"Remove these from your config. SkyRL handles them automatically.\n\n"
            f"FULL LIST OF SKYRL-INTERNAL KWARGS (never set these):\n{all_forbidden}\n\n"
            f"SAFE TO SET: custom_chat_template_*, kv_cache_dtype, quantization, cpu_offload_gb, etc."
        )


@dataclass
class ParsedRLConfig:
    """Result of parsing an RL configuration YAML file."""

    config_path: Path
    raw: Dict[str, Any]
    context_budget: ContextBudget
    entrypoint: str
    config_groups: Dict[str, str] = field(default_factory=dict)
    trainer: Dict[str, Any] = field(default_factory=dict)
    generator: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    terminal_bench: Optional[Dict[str, Any]] = None
    teacher: Optional[Dict[str, Any]] = None
    tensor_parallel_size: int = 1
    # "tasks" (default; terminal_bench task-dir extraction) or "parquet" (single-turn
    # RLVR: an HF id / .parquet is passed through to PromptDataset, NOT task-extracted).
    # Launcher-only (popped out of the `data` section so it never reaches Hydra).
    data_kind: str = "tasks"


@dataclass(frozen=True)
class ParsedCheckpointExportConfig:
    """Policy configuration needed to reconstruct a checkpoint for conversion."""

    config_path: Path
    config_groups: Dict[str, str]
    trainer: Dict[str, Any]


def validate_tp_divides_heads(
    tensor_parallel_size: int,
    num_attention_heads: Optional[int],
    config_path: Optional[Path] = None,
) -> None:
    """Fail fast if the inference TP size does not divide the model's attention-head count.

    vLLM requires ``num_attention_heads % tensor_parallel_size == 0``; a bad value (e.g.
    TP=8 against the dense delphi arch's 42 heads) wedges vLLM at engine init with no
    launcher-side signal. Skipped when ``num_attention_heads`` is unset (existing configs).

    Raises:
        ValueError: If ``num_attention_heads`` is set and not divisible by TP.
    """
    if not num_attention_heads:
        return
    if num_attention_heads % tensor_parallel_size != 0:
        valid = [t for t in range(1, num_attention_heads + 1) if num_attention_heads % t == 0]
        config_context = f" in {config_path}" if config_path else ""
        raise ValueError(
            f"generator.inference_engine_tensor_parallel_size={tensor_parallel_size} does not "
            f"divide model_num_attention_heads={num_attention_heads}{config_context}.\n"
            f"vLLM requires num_attention_heads % tensor_parallel_size == 0, or the engine "
            f"wedges at init with no launcher signal.\n"
            f"Valid TP values for {num_attention_heads} heads: {valid} (NEVER 8 for delphi's 42)."
        )


def resolve_rl_config_path(raw_path: str) -> Path:
    """Resolve an RL config path, checking the bundled ``configs/`` fallback.

    Resolution order: the path as-is, then ``SKYRL_CONFIG_DIR / raw_path``, then
    ``SKYRL_CONFIG_DIR / raw_path.yaml``.

    Raises:
        FileNotFoundError: If the config file cannot be found in any location.
    """
    path = Path(raw_path).expanduser()
    if path.exists():
        return path.resolve()

    fallback = SKYRL_CONFIG_DIR / raw_path
    if fallback.exists():
        return fallback.resolve()

    fallback_yaml = SKYRL_CONFIG_DIR / f"{raw_path}.yaml"
    if fallback_yaml.exists():
        return fallback_yaml.resolve()

    raise FileNotFoundError(
        f"RL config not found: {raw_path}\nSearched: {path}, {SKYRL_CONFIG_DIR / raw_path}, {fallback_yaml}"
    )


def materialize_rl_config(
    config_path: str,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Materialize a launcher-forwarded RL config inside the task container."""
    environment = os.environ if environment is None else environment
    payload = environment.get(RL_CONFIG_PAYLOAD_ENV)
    if payload is None:
        return config_path

    try:
        contents = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"Invalid base64 in {RL_CONFIG_PAYLOAD_ENV}") from error

    destination = Path(config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(contents)
    return str(destination)


def parse_rl_config(
    config_path: str,
    model_override: Optional[str] = None,
) -> ParsedRLConfig:
    """Parse an RL config YAML and extract all settings.

    Raises:
        FileNotFoundError: If config file cannot be found.
        yaml.YAMLError: If config file is not valid YAML.
    """
    path = resolve_rl_config_path(config_path)

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    context_budget = resolve_context_budget(raw, path)

    entrypoint = resolve_rl_entrypoint(raw.get("entrypoint"), config_path=path)
    config_groups = raw.get("config_groups", {})
    trainer, generator, terminal_bench, materialized_raw = _materialize_context_budget(raw, context_budget)
    data = dict(raw.get("data", {}))
    environment = raw.get("environment", {})
    teacher = raw.get("teacher")

    # data.kind is a launcher-only routing key (parquet vs. terminal_bench tasks); pop it
    # so it never leaks into the flattened Hydra args (SkyRL's `data` has no `kind` field).
    data_kind = data.pop("kind", "tasks")

    # Validate engine_init_kwargs doesn't contain SkyRL-internal keys.
    engine_init_kwargs = generator.get("engine_init_kwargs", {})
    validate_engine_init_kwargs(engine_init_kwargs, config_path=path)

    # Resolve relative paths in config sections to absolute paths so they work
    # regardless of the working directory at runtime. Skip data.train_data /
    # data.val_data as they may be HF repo IDs.
    trainer = resolve_paths_in_dict(trainer, skip_keys={"policy.model.path"})
    generator = resolve_paths_in_dict(generator)

    if model_override:
        trainer.setdefault("policy", {}).setdefault("model", {})["path"] = model_override

    tensor_parallel_size = generator.get("inference_engine_tensor_parallel_size", 1)

    # TP-divides-heads guard (delphi's 42-head arch forbids TP=8). No-op unless the config
    # declares model_num_attention_heads.
    validate_tp_divides_heads(tensor_parallel_size, raw.get("model_num_attention_heads"), config_path=path)

    return ParsedRLConfig(
        config_path=path,
        raw=materialized_raw,
        context_budget=context_budget,
        entrypoint=entrypoint,
        config_groups=config_groups,
        trainer=trainer,
        generator=generator,
        data=data,
        environment=environment,
        terminal_bench=terminal_bench,
        teacher=teacher,
        tensor_parallel_size=tensor_parallel_size,
        data_kind=data_kind,
    )


def parse_checkpoint_export_config(
    config_path: str,
    model_override: str,
) -> ParsedCheckpointExportConfig:
    """Read policy configuration without validating or materializing rollout settings."""
    path = resolve_rl_config_path(config_path)
    with path.open() as source:
        raw = yaml.safe_load(source) or {}

    trainer = resolve_paths_in_dict(copy.deepcopy(raw.get("trainer", {})), skip_keys={"policy.model.path"})
    trainer.setdefault("policy", {}).setdefault("model", {})["path"] = model_override
    return ParsedCheckpointExportConfig(
        config_path=path,
        config_groups=dict(raw.get("config_groups", {})),
        trainer=trainer,
    )


# Explicit mapping from custom environment import_paths to their base environment
# types. Used to determine tunnel requirements for custom environments.
IMPORT_PATH_TO_ENV_TYPE = {
    "harbor.environments.pooled.daytona_dind:PooledDaytonaDinDEnvironment": "daytona",
}


def extract_terminal_bench_agent_env(parsed: ParsedRLConfig) -> tuple:
    """Extract (agent_name, harbor_env) from a parsed terminal_bench config.

    Raises:
        ValueError: If import_path is specified but not in IMPORT_PATH_TO_ENV_TYPE.
    """
    tb = parsed.terminal_bench or {}
    harbor = tb.get("harbor", {})

    agent_name = harbor.get("name", "terminus-2")

    import_path = harbor.get("import_path")
    if import_path:
        if import_path not in IMPORT_PATH_TO_ENV_TYPE:
            raise ValueError(
                f"Unknown environment import_path: {import_path}\n"
                f"Add it to IMPORT_PATH_TO_ENV_TYPE in rl_config_translation.py.\n"
                f"Known import paths: {list(IMPORT_PATH_TO_ENV_TYPE.keys())}"
            )
        harbor_env = IMPORT_PATH_TO_ENV_TYPE[import_path]
    else:
        harbor_env = harbor.get("environment_type", "daytona")

    return agent_name, harbor_env


def _flatten_dict(d: Dict[str, Any], prefix: str = "", leaf_key_suffixes: tuple = ("rope_scaling",)) -> Dict[str, Any]:
    """Flatten a nested dictionary to dotted keys.

    Dicts whose key ends with a suffix in ``leaf_key_suffixes`` are kept as whole
    dict values rather than recursed into, so Hydra receives them as a single
    override.
    """
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and not any(k.endswith(s) for s in leaf_key_suffixes):
            items.update(_flatten_dict(v, key, leaf_key_suffixes))
        elif v is not None:
            items[key] = v
    return items


# Characters that require quoting in Hydra CLI values (special meaning in Hydra's
# override grammar or shell expansion).
HYDRA_SPECIAL_CHARS = frozenset("<>{}[]$`\\\"'=,()@#:*?!|;&\n\r\t ")


def _needs_quoting(s: str) -> bool:
    """Check if a string needs quoting for Hydra CLI."""
    return any(c in s for c in HYDRA_SPECIAL_CHARS)


def _quote_for_hydra(s: str) -> str:
    """Quote a string value for safe Hydra CLI passing."""
    # First escape backslashes, then newlines (order matters).
    escaped = s.replace("\\", "\\\\")
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    # For Hydra, wrap in single quotes and escape internal single quotes.
    escaped = escaped.replace("'", "'\\''")
    return f"'{escaped}'"


def format_hydra_arg(key: str, value: Any, prefix: str = "") -> str:
    """Format a single Hydra CLI argument.

    ``prefix`` selects the Hydra override mode: "" overrides an existing key, "+"
    adds a new key (fails if it exists), "++" adds-or-overrides.
    """
    if isinstance(value, bool):
        return f"{prefix}{key}={str(value).lower()}"
    elif isinstance(value, dict):
        # Format as a Hydra dict literal: {k1: v1, k2: v2}. Supports nested dicts.
        def _fmt_val(v: Any) -> str:
            if isinstance(v, bool):
                return str(v).lower()
            elif isinstance(v, dict):
                inner = ", ".join(f"{ik}: {_fmt_val(iv)}" for ik, iv in v.items())
                return f"{{{inner}}}"
            elif isinstance(v, (list, tuple)):
                items = ", ".join(_fmt_val(i) for i in v)
                return f"[{items}]"
            else:
                return str(v)

        dict_items = ", ".join(f"{k}: {_fmt_val(v)}" for k, v in value.items())
        return f"{prefix}{key}={{{dict_items}}}"
    elif isinstance(value, (list, tuple)):
        # Format as a YAML list WITHOUT outer quotes so Hydra parses it as a list.
        # Double-quote string items to handle paths with special chars.
        items = ",".join(f'"{v}"' if isinstance(v, str) else str(v) for v in value)
        return f"{prefix}{key}=[{items}]"
    elif isinstance(value, str):
        if _needs_quoting(value):
            return f"{prefix}{key}={_quote_for_hydra(value)}"
        else:
            return f"{prefix}{key}={value}"
    else:
        return f"{prefix}{key}={value}"


_OPTIONAL_HYDRA_PATTERNS = {
    ".engine_init_kwargs",
    ".hf_hub_",
    ".enable_db_registration",
    ".optimizer_kwargs",
    ".rope_scaling",
    ".wrap_policy",
    ".transformer_config_kwargs",
}


def _apply_policy_model_source(trainer: Dict[str, Any], exp_args: Dict[str, Any]) -> str | None:
    """Apply the task-visible policy path and its replayable source identity."""
    model_path = exp_args.get("model_path")
    if not model_path:
        return None
    policy_model = trainer.setdefault("policy", {}).setdefault("model", {})
    policy_model["path"] = model_path
    model_source = model_source_for_path(
        model_path,
        exp_args.get("model_source_uri"),
        exp_args.get("model_source_identity"),
    )
    if model_source:
        policy_model["source_uri"] = model_source.uri
        policy_model["source_identity"] = model_source.identity
    return model_path


def _role_gpus_per_node(
    placement: Dict[str, Any],
    key: str,
    launch_gpus_per_node: int,
    *,
    preserve_smaller_value: bool,
) -> int:
    configured = placement.get(key)
    if preserve_smaller_value and configured is not None and int(configured) <= launch_gpus_per_node:
        return int(configured)
    return launch_gpus_per_node


def build_checkpoint_export_hydra_args(
    parsed: ParsedCheckpointExportConfig,
    exp_args: Dict[str, Any],
    hpc: HPCGeometry,
) -> List[str]:
    """Build policy-only Hydra arguments for the standalone checkpoint converter."""
    args = [f"+{group_name}={config_name}" for group_name, config_name in parsed.config_groups.items()]
    trainer = copy.deepcopy(parsed.trainer)
    placement = trainer.setdefault("placement", {})
    num_nodes = int(exp_args.get("num_nodes", 1))
    gpus_per_node = int(exp_args.get("gpus_per_node", hpc.gpus_per_node))
    placement["policy_num_nodes"] = num_nodes
    placement["policy_num_gpus_per_node"] = _role_gpus_per_node(
        placement,
        "policy_num_gpus_per_node",
        gpus_per_node,
        preserve_smaller_value=False,
    )
    _apply_policy_model_source(trainer, exp_args)

    for key, value in _flatten_dict(trainer, "trainer").items():
        prefix = "++" if any(pattern in key for pattern in _OPTIONAL_HYDRA_PATTERNS) else ""
        args.append(format_hydra_arg(key, value, prefix=prefix))
    return args


def _apply_trajectory_retention_path(generator: Dict[str, Any], experiments_dir: str, job_name: str) -> None:
    retention = dict(generator.get("trajectory_retention", {}))
    configured_path = retention.get("output_path")
    if not configured_path and experiments_dir and job_name:
        retention["output_path"] = join_resource_path(experiments_dir, job_name, "trace_jobs", "training_trajectories")
    if retention:
        generator["trajectory_retention"] = retention


def build_skyrl_hydra_args(
    parsed: ParsedRLConfig,
    exp_args: Dict[str, Any],
    hpc: HPCGeometry,
) -> List[str]:
    """Convert a parsed config + exp_args into Hydra CLI argument strings.

    Adds config groups, derives paths from experiments_dir/job_name, computes
    num_inference_engines from the cluster config, flattens nested dicts to dotted
    Hydra keys, and applies data paths from the CLI.
    """
    args = []

    # Config groups (+ prefix for Hydra).
    for group_name, config_name in parsed.config_groups.items():
        args.append(f"+{group_name}={config_name}")

    # Make copies to avoid mutating parsed config.
    trainer = dict(parsed.trainer)
    generator = dict(parsed.generator)
    data = dict(parsed.data)
    environment = dict(parsed.environment)

    # Derive paths if null.
    experiments_dir = exp_args.get("experiments_dir", "")
    job_name = exp_args.get("job_name", "")

    if not trainer.get("run_name") and job_name:
        trainer["run_name"] = job_name
    if not trainer.get("export_path") and experiments_dir and job_name:
        trainer["export_path"] = join_resource_path(experiments_dir, job_name, "exports")
        print(f"Auto-set trainer.export_path: {trainer['export_path']}")
    if not trainer.get("ckpt_path") and experiments_dir and job_name:
        trainer["ckpt_path"] = join_resource_path(experiments_dir, job_name, "checkpoints")
        print(f"Auto-set trainer.ckpt_path: {trainer['ckpt_path']}")
    _apply_trajectory_retention_path(generator, experiments_dir, job_name)

    # Derive placement from num_nodes.
    num_nodes = int(exp_args.get("num_nodes", 1))
    gpus_per_node = int(exp_args.get("gpus_per_node", hpc.gpus_per_node))
    placement = dict(trainer.get("placement", {}))

    policy_num_nodes = exp_args.get("policy_num_nodes")
    if placement.get("policy_num_nodes") is None:
        placement["policy_num_nodes"] = policy_num_nodes if policy_num_nodes is not None else num_nodes
    if placement.get("ref_num_nodes") is None:
        placement["ref_num_nodes"] = policy_num_nodes if policy_num_nodes is not None else num_nodes

    # Derive gpus_per_node from the CLI (cluster-specific, not hardcoded in YAML).
    # EXCEPTION (rank-spread lever): honor an EXPLICIT YAML *_num_gpus_per_node when
    # it is <= the node's gpus_per_node, so policy/ref can use FEWER GPUs per
    # (reserved whole) node than the node physically has — spreading a fixed
    # policy-rank count over MORE nodes.
    def _resolve_gpus_per_node(key: str) -> int:
        return _role_gpus_per_node(
            placement,
            key,
            gpus_per_node,
            preserve_smaller_value=True,
        )

    placement["policy_num_gpus_per_node"] = _resolve_gpus_per_node("policy_num_gpus_per_node")
    placement["ref_num_gpus_per_node"] = _resolve_gpus_per_node("ref_num_gpus_per_node")
    trainer["placement"] = placement

    # Compute num_inference_engines.
    tp_size = parsed.tensor_parallel_size
    if generator.get("num_inference_engines") is None:
        generator["num_inference_engines"] = (num_nodes * gpus_per_node) // tp_size

    # Data paths from CLI.
    if exp_args.get("train_data"):
        train_data = exp_args["train_data"]
        if isinstance(train_data, str) and train_data.startswith("["):
            import ast

            try:
                train_data = ast.literal_eval(train_data)
            except (ValueError, SyntaxError):
                pass
        data["train_data"] = train_data

    if exp_args.get("val_data"):
        val_data = exp_args["val_data"]
        if isinstance(val_data, str) and val_data.startswith("["):
            import ast

            try:
                val_data = ast.literal_eval(val_data)
            except (ValueError, SyntaxError):
                pass
        data["val_data"] = val_data

    # Model path and served_model_name for Harbor/LiteLLM compatibility.
    model_path = _apply_policy_model_source(trainer, exp_args)
    if model_path:
        # served_model_name: strip the org prefix from "org/model" HF IDs, since
        # Harbor/LiteLLM requires model names with exactly one '/'.
        served_model_name = model_path.split("/")[-1] if "/" in model_path else model_path
        generator.setdefault("engine_init_kwargs", {})["served_model_name"] = served_model_name

    # HuggingFace Hub upload settings. Default to laion/<job_name> if not provided.
    hf_hub_repo_id = exp_args.get("hf_hub_repo_id")
    if not hf_hub_repo_id and job_name:
        hf_hub_repo_id = f"laion/{job_name}"
        print(f"HF Hub upload auto-defaulted to: {hf_hub_repo_id}")
    if hf_hub_repo_id:
        trainer["hf_hub_repo_id"] = hf_hub_repo_id
        if exp_args.get("hf_hub_repo_id"):
            print(f"HF Hub upload enabled: {hf_hub_repo_id}")
    hf_hub_private = exp_args.get("hf_hub_private", False)
    if hf_hub_private:
        trainer["hf_hub_private"] = True

    # Trace upload CLI overrides (apply to terminal_bench.trace_upload).
    if parsed.terminal_bench is not None:
        trace_upload = parsed.terminal_bench.setdefault("trace_upload", {})
        if exp_args.get("trace_upload_enabled") is not None:
            trace_upload["enabled"] = exp_args["trace_upload_enabled"]
        if exp_args.get("trace_upload_repo_org"):
            trace_upload["repo_org"] = exp_args["trace_upload_repo_org"]
        if exp_args.get("trace_upload_episodes"):
            trace_upload["episodes"] = exp_args["trace_upload_episodes"]
        if exp_args.get("trace_upload_dataset_type"):
            trace_upload["dataset_type"] = exp_args["trace_upload_dataset_type"]
        if exp_args.get("trace_upload_cleanup") is not None:
            trace_upload["cleanup"] = exp_args["trace_upload_cleanup"]

    # Build args for each section. Keys under these patterns may not exist in
    # SkyRL's base config, so use the ++ prefix (add-or-override):
    #   engine_init_kwargs, hf_hub_*, enable_db_registration, optimizer_kwargs,
    #   rope_scaling, wrap_policy, transformer_config_kwargs.
    # transformer_config_kwargs is a passthrough to Megatron's TransformerConfig
    # (megatron_worker setattr's each subkey), so it may carry keys NOT declared in the
    # base preset (e.g. gradient_accumulation_fusion). On megatron-bridge 0.5.0 (#33/#34)
    # that node became a strict struct, so a plain "" override of a new subkey fails
    # ("Could not override ...transformer_config_kwargs.gradient_accumulation_fusion");
    # ++ force-adds the leaf while leaving the preset's other subkeys (recompute_*) intact.
    for section, values in [
        ("trainer", trainer),
        ("generator", generator),
        ("data", data),
        ("environment", environment),
    ]:
        for key, val in _flatten_dict(values, section).items():
            prefix = "++" if any(pattern in key for pattern in _OPTIONAL_HYDRA_PATTERNS) else ""
            args.append(format_hydra_arg(key, val, prefix=prefix))

    # Teacher config (on-policy distillation) — all keys use ++ since the teacher
    # section doesn't exist in SkyRL's base Hydra config.
    if parsed.teacher:
        for key, val in _flatten_dict(parsed.teacher, "teacher").items():
            args.append(format_hydra_arg(key, val, prefix="++"))

    # Terminal-Bench experiments may override packaged group keys or add new ones.
    if parsed.terminal_bench:
        terminal_bench = dict(parsed.terminal_bench)

        # Derive trials_dir from experiments_dir if not set.
        if not terminal_bench.get("trials_dir") and experiments_dir and job_name:
            terminal_bench["trials_dir"] = f"{experiments_dir}/{job_name}/trace_jobs"

        for key, val in _flatten_dict(terminal_bench).items():
            args.append(format_hydra_arg(f"terminal_bench_config.{key}", val, prefix="++"))

    return args


def get_skyrl_command_preview(
    entrypoint: str,
    hydra_args: List[str],
    max_args_shown: int = 10,
) -> str:
    """Generate a preview of the SkyRL command for dry-run output."""
    lines = [f"python -m {entrypoint} \\"]

    for i, arg in enumerate(hydra_args):
        if i < max_args_shown:
            lines.append(f"  {arg} \\")
        elif i == max_args_shown:
            lines.append(f"  ... ({len(hydra_args) - max_args_shown} more arguments)")
            break

    if lines and lines[-1].endswith(" \\"):
        lines[-1] = lines[-1][:-2]

    return "\n".join(lines)
