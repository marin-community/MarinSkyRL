"""RL training configuration parsing + Hydra-argument translation for MarinSkyRL.

Provides YAML-based configuration for SkyRL RL training jobs, replacing 50+ Hydra
CLI arguments with a single ``--rl_config`` YAML file.

Usage::

    from cloud.iris.rl_config_translation import parse_rl_config, build_skyrl_hydra_args

    parsed = parse_rl_config("configs/56gpu_qwen3_8b.yaml")
    hydra_args = build_skyrl_hydra_args(parsed, exp_args, hpc)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from cloud.iris.paths import resolve_paths_in_dict

# Directory containing the bundled example RL config YAML files.
SKYRL_CONFIG_DIR = Path(__file__).parent / "configs"

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
    entrypoint: str
    config_groups: Dict[str, str] = field(default_factory=dict)
    trainer: Dict[str, Any] = field(default_factory=dict)
    generator: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    terminal_bench: Optional[Dict[str, Any]] = None
    teacher: Optional[Dict[str, Any]] = None
    tensor_parallel_size: int = 1


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

    entrypoint = raw.get("entrypoint", "skyrl_train.entrypoints.main_base")
    config_groups = raw.get("config_groups", {})
    trainer = raw.get("trainer", {})
    generator = raw.get("generator", {})
    data = raw.get("data", {})
    terminal_bench = raw.get("terminal_bench")
    teacher = raw.get("teacher")

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

    return ParsedRLConfig(
        config_path=path,
        raw=raw,
        entrypoint=entrypoint,
        config_groups=config_groups,
        trainer=trainer,
        generator=generator,
        data=data,
        terminal_bench=terminal_bench,
        teacher=teacher,
        tensor_parallel_size=tensor_parallel_size,
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


def _format_hydra_arg(key: str, value: Any, prefix: str = "") -> str:
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


def build_skyrl_hydra_args(
    parsed: ParsedRLConfig,
    exp_args: Dict[str, Any],
    hpc: Any,
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

    # Derive paths if null.
    experiments_dir = exp_args.get("experiments_dir", "")
    job_name = exp_args.get("job_name", "")

    if not trainer.get("run_name") and job_name:
        trainer["run_name"] = job_name
    if not trainer.get("export_path") and experiments_dir and job_name:
        trainer["export_path"] = f"{experiments_dir}/{job_name}/exports"
        print(f"Auto-set trainer.export_path: {trainer['export_path']}")
    if not trainer.get("ckpt_path") and experiments_dir and job_name:
        trainer["ckpt_path"] = f"{experiments_dir}/{job_name}/checkpoints"
        print(f"Auto-set trainer.ckpt_path: {trainer['ckpt_path']}")

    # Derive placement from num_nodes.
    num_nodes = int(exp_args.get("num_nodes", 1))
    gpus_per_node = int(exp_args.get("gpus_per_node", getattr(hpc, "gpus_per_node", 4)))
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
        yaml_val = placement.get(key)
        if yaml_val is None:
            return gpus_per_node
        if exp_args.get("gpus_per_node") and int(yaml_val) > gpus_per_node:
            # A YAML value LARGER than the node has is a mis-size; clamp to CLI.
            return gpus_per_node
        return int(yaml_val)

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
    model_path = exp_args.get("model_path")
    if model_path:
        trainer.setdefault("policy", {}).setdefault("model", {})["path"] = model_path

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
    optional_patterns = {
        ".engine_init_kwargs",
        ".hf_hub_",
        ".enable_db_registration",
        ".optimizer_kwargs",
        ".rope_scaling",
        ".wrap_policy",
        ".transformer_config_kwargs",
    }

    for section, values in [("trainer", trainer), ("generator", generator), ("data", data)]:
        for key, val in _flatten_dict(values, section).items():
            prefix = "++" if any(pattern in key for pattern in optional_patterns) else ""
            args.append(_format_hydra_arg(key, val, prefix=prefix))

    # Teacher config (on-policy distillation) — all keys use ++ since the teacher
    # section doesn't exist in SkyRL's base Hydra config.
    if parsed.teacher:
        for key, val in _flatten_dict(parsed.teacher, "teacher").items():
            args.append(_format_hydra_arg(key, val, prefix="++"))

    # Terminal bench with + prefix (new keys added by the config group).
    if parsed.terminal_bench:
        terminal_bench = dict(parsed.terminal_bench)

        # Derive trials_dir from experiments_dir if not set.
        if not terminal_bench.get("trials_dir") and experiments_dir and job_name:
            terminal_bench["trials_dir"] = f"{experiments_dir}/{job_name}/trace_jobs"

        for key, val in _flatten_dict(terminal_bench).items():
            args.append(_format_hydra_arg(f"terminal_bench_config.{key}", val, prefix="+"))

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
