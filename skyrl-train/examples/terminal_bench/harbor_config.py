"""
Schema-driven Harbor configuration mapping for SkyRL terminal bench.

This module provides automatic mapping from YAML config to Harbor's TrialConfig,
with validation and warnings for unknown/unsupported fields.

Usage:
    from examples.terminal_bench.harbor_config import HarborConfigBuilder

    builder = HarborConfigBuilder(terminal_bench_cfg)
    trial_config = builder.build_trial_config(
        task_path=prompt,
        trials_dir=self.trials_dir,
        model_name="hosted_vllm/Qwen3-8B",
        api_base="http://localhost:8000/v1",
        session_id=session_id,
    )

Agent name is now read from the harbor config section (defaults to "terminus-2"):
    terminal_bench:
      harbor:
        name: terminus-2  # Harbor AgentName value
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from harbor.models.trial.config import (
    TrialConfig,
    AgentConfig,
    TaskConfig,
    EnvironmentConfig,
    VerifierConfig,
)
from harbor.models.job.config import RetryConfig
from harbor.models.environment_type import EnvironmentType
from harbor.models.agent.name import AgentName


# =============================================================================
# Schema Definition: Which Harbor fields are exposed in SkyRL YAML
# =============================================================================
#
# This schema defines the mapping between YAML config keys and Harbor's
# Pydantic models. To expose a new Harbor field:
#   1. Add it to the appropriate section below
#   2. That's it - the mapping is automatic
#
# Field types:
#   - "direct": Maps directly to a Pydantic model field
#   - "kwargs": Passed through agent.kwargs dict (agent-specific params)
#
# =============================================================================


@dataclass
class FieldMapping:
    """Defines how a YAML field maps to Harbor config."""

    harbor_field: str  # Field name in Harbor's Pydantic model
    field_type: str = "direct"  # "direct" or "kwargs"
    default: Any = None  # Default value if not specified


@dataclass
class SectionSchema:
    """Schema for a Harbor config section (agent, environment, etc.)."""

    fields: Dict[str, FieldMapping] = field(default_factory=dict)

    def get_all_field_names(self) -> Set[str]:
        return set(self.fields.keys())


# Agent config fields
AGENT_SCHEMA = SectionSchema(
    fields={
        # Direct fields on AgentConfig
        "name": FieldMapping("name", default="terminus-2"),  # Maps to AgentConfig.name (Harbor AgentName)
        "override_timeout_sec": FieldMapping("override_timeout_sec"),
        "override_setup_timeout_sec": FieldMapping("override_setup_timeout_sec"),
        "max_timeout_sec": FieldMapping("max_timeout_sec"),
        # Kwargs fields (passed to agent.kwargs)
        "max_episodes": FieldMapping("max_episodes", field_type="kwargs", default=16),
        "enable_summarize": FieldMapping("enable_summarize", field_type="kwargs", default=True),
        "store_all_messages": FieldMapping("store_all_messages", field_type="kwargs", default=True),
        # Thinking/reasoning settings
        "interleaved_thinking": FieldMapping("interleaved_thinking", field_type="kwargs", default=False),
        # Extra body params passed to LLM API (e.g., chat_template_kwargs for enable_thinking)
        "extra_body": FieldMapping("extra_body", field_type="kwargs"),
        # Rollout details collection (for TIS in async training)
        # When true, collects per-token logprobs needed for importance sampling correction
        "collect_rollout_details": FieldMapping("collect_rollout_details", field_type="kwargs", default=False),
        # Strict JSON parser mode (for RL training)
        # When true, treats parser warnings as errors and disables auto-correction.
        # This prevents reward hacking where the model produces garbage output that the
        # parser auto-corrects, allowing the model to get rewards despite malformed responses.
        "strict_json_parser": FieldMapping("strict_json_parser", field_type="kwargs", default=False),
        # Episode logging control
        # When false, disables creation of episode-* folders with debug.json, prompt.txt, response.txt
        # This reduces disk I/O for RL training where SkyRL uses TrialResult directly
        "enable_episode_logging": FieldMapping("enable_episode_logging", field_type="kwargs", default=True),
        # Terminal session recording (asciinema)
        # When false, disables recording.cast file generation
        # This significantly reduces disk I/O for RL training
        "record_terminal_session": FieldMapping("record_terminal_session", field_type="kwargs", default=True),
        # Pane logging control
        # When false, disables terminus_2.pane file generation
        # This reduces disk I/O for RL training
        "enable_pane_logging": FieldMapping("enable_pane_logging", field_type="kwargs", default=True),
        # Trajectory configuration (dict passed as trajectory_config kwarg to Terminus-2)
        # raw_content: If True, save raw LLM responses (including <think> blocks) in trajectory
        # linear_history: If True, split trajectory into separate files on summarization
        "trajectory_config": FieldMapping("trajectory_config", field_type="kwargs"),
    }
)

# Eval-specific config fields
# These settings override the standard settings during evaluation
EVAL_SCHEMA = SectionSchema(
    fields={
        # Timeout override for eval (default 900s = 15 minutes)
        # Eval tasks may need more time than training since we don't retry
        "eval_timeout_override_sec": FieldMapping("eval_timeout_override_sec", default=900),
    }
)

# Environment config fields
ENVIRONMENT_SCHEMA = SectionSchema(
    fields={
        "override_cpus": FieldMapping("override_cpus"),
        "override_memory_mb": FieldMapping("override_memory_mb"),
        "override_storage_mb": FieldMapping("override_storage_mb"),
        "override_gpus": FieldMapping("override_gpus"),
        "environment_type": FieldMapping("type"),  # Maps to EnvironmentConfig.type
        "import_path": FieldMapping("import_path"),  # Custom environment class
        # Pool-based environment kwargs (for PooledDaytonaDinDEnvironment)
        "pool_size": FieldMapping("pool_size", field_type="kwargs"),
        "acquire_timeout": FieldMapping("acquire_timeout", field_type="kwargs"),
        "env_cpu": FieldMapping("cpu", field_type="kwargs"),  # env_ prefix to avoid conflict with override_cpus
        "env_memory_gb": FieldMapping("memory_gb", field_type="kwargs"),
        "env_disk_gb": FieldMapping("disk_gb", field_type="kwargs"),
        # Daytona snapshot support (for reducing rate limits with many sandboxes)
        # auto_snapshot: Automatically create/reuse snapshots based on Dockerfile hash
        # snapshot_template_name: Use explicit snapshot name template (e.g., "harbor__{name}__snapshot")
        "auto_snapshot": FieldMapping("auto_snapshot", field_type="kwargs", default=False),
        "snapshot_template_name": FieldMapping("snapshot_template_name", field_type="kwargs"),
    }
)

# Verifier config fields
VERIFIER_SCHEMA = SectionSchema(
    fields={
        "verifier_override_timeout_sec": FieldMapping("override_timeout_sec"),
        "verifier_max_timeout_sec": FieldMapping("max_timeout_sec"),
        "verifier_disable": FieldMapping("disable"),
    }
)

# Trial-level config fields
TRIAL_SCHEMA = SectionSchema(
    fields={
        "timeout_multiplier": FieldMapping("timeout_multiplier", default=1.0),
    }
)

# Retry config fields (for QueueOrchestrator)
RETRY_SCHEMA = SectionSchema(
    fields={
        "max_retries": FieldMapping("max_retries", default=2),
        "min_wait_sec": FieldMapping("min_wait_sec", default=1.0),
        "max_wait_sec": FieldMapping("max_wait_sec", default=60.0),
        "wait_multiplier": FieldMapping("wait_multiplier", default=2.0),
        # Exception filtering - comma-separated strings in YAML, converted to sets
        "include_exceptions": FieldMapping("include_exceptions"),
        "exclude_exceptions": FieldMapping("exclude_exceptions"),
    }
)

# Orchestrator config fields
ORCHESTRATOR_SCHEMA = SectionSchema(
    fields={
        "n_concurrent_trials": FieldMapping("n_concurrent_trials"),
    }
)

# Logging config fields
LOGGING_SCHEMA = SectionSchema(
    fields={
        # Log level for Harbor (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        "log_level": FieldMapping("log_level", default="WARNING"),
    }
)

# Reward shaping config fields
REWARD_SHAPING_SCHEMA = SectionSchema(
    fields={
        # Parser for test output (pytest, unittest, generic, or None for auto-detect)
        "reward_parser": FieldMapping("reward_parser", default=None),
        # Shaper strategy:
        #   Verifier-based: pass_ratio, effective_pass_ratio, weighted, threshold, binary_partial, original
        #   Trajectory-based: thinking_length, format_quality
        #   Composite: composite (weighted combination of verifier + trajectory shapers)
        "reward_shaper": FieldMapping("reward_shaper", default="pass_ratio"),
        # Whether to enable reward shaping (if False, uses original binary reward)
        "enable_reward_shaping": FieldMapping("enable_reward_shaping", default=False),
        # Fallback to original reward if parsing fails
        "reward_shaping_fallback": FieldMapping("reward_shaping_fallback", default=True),
        # Threshold shaper params
        "reward_threshold": FieldMapping("reward_threshold", default=1.0),
        "below_threshold_scale": FieldMapping("below_threshold_scale", default=0.5),
        # Binary partial shaper params
        "partial_threshold": FieldMapping("partial_threshold", default=0.9),
        "partial_credit": FieldMapping("partial_credit", default=0.5),
        # Thinking length shaper params
        "thinking_target_tokens": FieldMapping("thinking_target_tokens", default=750),
        "thinking_sigma_tokens": FieldMapping("thinking_sigma_tokens", default=250),
        "thinking_min_turns_ratio": FieldMapping("thinking_min_turns_ratio", default=0.5),
        # Format quality shaper params
        "format_required_fields": FieldMapping("format_required_fields", default=None),
        "format_penalize_truncated": FieldMapping("format_penalize_truncated", default=True),
        # Command quality shaper params
        "command_quality_error_penalty_weight": FieldMapping("command_quality_error_penalty_weight", default=1.0),
        "command_quality_min_turns": FieldMapping("command_quality_min_turns", default=2),
        # Composite shaper params
        "composite_components": FieldMapping("composite_components", default=None),
        "composite_verifier_shaper": FieldMapping("composite_verifier_shaper", default="pass_ratio"),
        # composite_loop (loop-behavior reward shaping) params.
        # `loop_shaping` is the nested config block (all components default-off);
        # see skyrl_train.utils.reward_shaping.DEFAULT_LOOP_SHAPING_CONFIG.
        # `loop_outcome_shaper` selects which verifier-based shaper computes the
        # outcome term (default pass_ratio -> byte-identical to today when the
        # components are off). Both default to None so the composite_loop shaper
        # falls back to its own defaults when unset.
        "loop_shaping": FieldMapping("loop_shaping", default=None),
        "loop_outcome_shaper": FieldMapping("loop_outcome_shaper", default=None),
        # Loop-behavior reward shaping (Stage B / F5 + F4): the master gate for the
        # per-token shaping channel + span tagger. Default False -> the generator
        # emits NEITHER token_level_shaping NOR response_span_tags, so the
        # GeneratorOutput is byte-identical to today. When True, Stage B emits the
        # channel as ZEROS (no-op) plus the span tags; Stages C/D fill the channel.
        "enable_token_reward_channel": FieldMapping("enable_token_reward_channel", default=False),
        # When True (and the channel is enabled) also emit F4 span tags. Separable
        # so the tagger cost can be disabled independently. No-op when the channel
        # is off.
        "enable_span_tagging": FieldMapping("enable_span_tagging", default=True),
    }
)

# Error handling config fields (for RLOO-N advantage estimator)
# Controls how different failure types are treated:
# - "mask" exceptions: Excluded from baseline (neutral - infrastructure failures)
# - "zero" exceptions: Included in baseline with reward=0 (agent failures)
#
# Default classification:
# - Infrastructure failures (mask): DaytonaError, NetworkError, EnvironmentStartTimeoutError
# - Agent failures (zero): AgentTimeoutError, ContextLengthExceededError
# - Ambiguous (configurable): VerifierTimeoutError, RewardFileNotFoundError
ERROR_HANDLING_SCHEMA = SectionSchema(
    fields={
        # Enable RLOO-N style error handling (exclude infrastructure failures from baseline)
        "enable_error_classification": FieldMapping("enable_error_classification", default=False),
        # Exceptions to pass through (ignore exception, use verifier reward normally).
        # The verifier still runs after these errors in Harbor, so we get a real reward.
        # Use for soft limits like timeout/context-length where partial work is evaluated.
        "passthrough_exceptions": FieldMapping("passthrough_exceptions", default=[]),
        # Exceptions to mask (exclude from baseline, no gradient contribution)
        # These are treated as "neutral" - infrastructure issues, not agent failures
        "mask_exceptions": FieldMapping(
            "mask_exceptions",
            default=[
                "DaytonaError",
                "EnvironmentStartTimeoutError",
                "NetworkError",
                "ConnectionError",
                "RewardFileNotFoundError",
                "RewardFileEmptyError",
            ],
        ),
        # Exceptions to zero (include in baseline with reward=0)
        # These are treated as agent failures - the model should learn to avoid them
        "zero_exceptions": FieldMapping(
            "zero_exceptions",
            default=[
                "AgentTimeoutError",
                "ContextLengthExceededError",
            ],
        ),
        # Default treatment for unclassified exceptions ("mask", "zero", or "passthrough")
        "default_error_treatment": FieldMapping("default_error_treatment", default="zero"),
    }
)

# Complete schema registry
HARBOR_SCHEMA = {
    "agent": AGENT_SCHEMA,
    "environment": ENVIRONMENT_SCHEMA,
    "verifier": VERIFIER_SCHEMA,
    "trial": TRIAL_SCHEMA,
    "retry": RETRY_SCHEMA,
    "orchestrator": ORCHESTRATOR_SCHEMA,
    "logging": LOGGING_SCHEMA,
    "reward_shaping": REWARD_SHAPING_SCHEMA,
    "error_handling": ERROR_HANDLING_SCHEMA,
    "eval": EVAL_SCHEMA,
}


def _get_all_known_harbor_fields() -> Set[str]:
    """Get all field names that Harbor's Pydantic models accept."""
    known = set()
    # From AgentConfig
    known.update(AgentConfig.model_fields.keys())
    # From EnvironmentConfig
    known.update(EnvironmentConfig.model_fields.keys())
    # From VerifierConfig
    known.update(VerifierConfig.model_fields.keys())
    # From TrialConfig (excluding nested configs)
    known.update({"timeout_multiplier", "trial_name"})
    return known


def _get_all_exposed_fields() -> Set[str]:
    """Get all field names exposed in our schema."""
    exposed = set()
    for schema in HARBOR_SCHEMA.values():
        exposed.update(schema.get_all_field_names())
    return exposed


# =============================================================================
# HarborConfigBuilder: Main interface for building TrialConfig from YAML
# =============================================================================


class HarborConfigBuilder:
    """
    Builds Harbor TrialConfig from SkyRL YAML configuration.

    Provides automatic field mapping with validation and warnings.
    """

    def __init__(self, terminal_bench_cfg: DictConfig):
        """
        Initialize the builder with terminal bench configuration.

        Args:
            terminal_bench_cfg: The terminal_bench_config section from Hydra config.
        """
        self._cfg = terminal_bench_cfg
        self._warnings_issued: Set[str] = set()

        # Extract harbor-specific config if present, otherwise use flat structure
        # This supports both new nested style and legacy flat style
        if "harbor" in terminal_bench_cfg:
            self._harbor_cfg = OmegaConf.to_container(terminal_bench_cfg.harbor, resolve=True) or {}
        else:
            # Legacy: extract harbor fields from flat config
            self._harbor_cfg = self._extract_harbor_fields_legacy(terminal_bench_cfg)

        # Extract model_info (special handling - nested dict passed to agent kwargs)
        model_info_cfg = terminal_bench_cfg.get("model_info", {})
        if isinstance(model_info_cfg, DictConfig):
            model_info_cfg = OmegaConf.to_container(model_info_cfg, resolve=True)
        self._model_info = {
            "max_input_tokens": model_info_cfg.get("max_input_tokens", 32768),
            "max_output_tokens": model_info_cfg.get("max_output_tokens", 8192),
            "input_cost_per_token": model_info_cfg.get("input_cost_per_token", 0),
            "output_cost_per_token": model_info_cfg.get("output_cost_per_token", 0),
        }

        # Extract PRM (Process Reward Model) config for mid-trial thrashing detection.
        # When enabled, the PRM's should_terminate() is passed as turn_callback to
        # terminus-2, allowing early termination of thrashing agents.
        self._turn_callback = self._build_prm_turn_callback()

        # Validate config and issue warnings
        self._validate_config()

    def _build_prm_turn_callback(self):
        """Build a PRM turn_callback from the ``prm`` config section.

        Reads from ``terminal_bench.harbor.prm`` (the standard location, already
        normalized into ``self._harbor_cfg``). Older flat-config layouts that
        had ``terminal_bench.prm`` at the top level are no longer supported —
        every prod config nests ``prm`` under ``harbor``.

        Returns:
            A callable ``(turn, trajectory_steps, messages) -> bool`` if a PRM
            is configured, or ``None`` if disabled (``prm.name`` is null/empty).
        """
        prm_cfg = self._harbor_cfg.get("prm", None)
        if prm_cfg is None:
            return None

        # ``self._harbor_cfg`` is already a plain dict (OmegaConf.to_container'd
        # in __init__), but be defensive in case a future caller passes a
        # DictConfig in directly.
        if isinstance(prm_cfg, DictConfig):
            prm_cfg = OmegaConf.to_container(prm_cfg, resolve=True)

        prm_name = prm_cfg.get("name", None)
        if not prm_name:
            return None

        # Lazy import so the prm package is only required when actually used.
        from prm import get_prm  # noqa: E402

        prm_kwargs = {k: v for k, v in prm_cfg.items() if k != "name"}
        prm_instance = get_prm(prm_name, **prm_kwargs)
        logger.info(f"PRM '{prm_name}' enabled as turn_callback " f"(params: {prm_kwargs})")
        return prm_instance.as_turn_callback()

    def _extract_harbor_fields_legacy(self, cfg: DictConfig) -> Dict[str, Any]:
        """Extract harbor-related fields from legacy flat config structure."""
        harbor_fields = {}
        all_exposed = _get_all_exposed_fields()

        for key in all_exposed:
            if key in cfg and cfg[key] is not None:
                harbor_fields[key] = cfg[key]

        return harbor_fields

    # Harbor-block keys that are SkyRL extensions handled outside HARBOR_SCHEMA
    # (so the schema validator doesn't flag them as "unknown"). PRM lives here
    # because it's consumed directly by _build_prm_turn_callback to construct
    # a turn_callback, not by the schema-based field-mapping pipeline.
    SKYRL_EXTENSION_KEYS = frozenset({"prm"})

    def _validate_config(self) -> None:
        """Validate config and issue warnings for unknown/unsupported fields."""
        all_exposed = _get_all_exposed_fields()
        all_known_harbor = _get_all_known_harbor_fields()

        for key, value in self._harbor_cfg.items():
            if value is None:
                continue

            if key in self.SKYRL_EXTENSION_KEYS:
                # Handled by a dedicated builder (e.g. _build_prm_turn_callback);
                # not part of the schema-driven field mapping.
                continue

            if key not in all_exposed:
                if key in all_known_harbor:
                    # Known Harbor field but not exposed in SkyRL
                    self._warn_once(
                        f"Harbor config '{key}' is a valid Harbor field but not exposed "
                        f"in SkyRL. Add to HARBOR_SCHEMA in harbor_config.py to enable."
                    )
                else:
                    # Completely unknown field
                    self._warn_once(
                        f"Unknown harbor config key '{key}' - ignoring. "
                        f"Check spelling or Harbor version compatibility."
                    )

    def _warn_once(self, message: str) -> None:
        """Issue a warning only once per message."""
        if message not in self._warnings_issued:
            self._warnings_issued.add(message)
            logger.warning(message)
            warnings.warn(message, UserWarning, stacklevel=3)

    def _get_field_value(
        self,
        yaml_key: str,
        mapping: FieldMapping,
        fallback_cfg: Optional[DictConfig] = None,
    ) -> Any:
        """Get field value from config with fallback to default."""
        # Check harbor config first
        if yaml_key in self._harbor_cfg:
            return self._harbor_cfg[yaml_key]

        # Check fallback (legacy flat config)
        if fallback_cfg is not None and yaml_key in fallback_cfg:
            value = fallback_cfg.get(yaml_key)
            if value is not None:
                return value

        # Return default
        return mapping.default

    def _build_agent_fields(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build agent direct fields and kwargs from config."""
        direct_fields = {}
        kwargs_fields = {}

        for yaml_key, mapping in AGENT_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                if mapping.field_type == "kwargs":
                    kwargs_fields[mapping.harbor_field] = value
                else:
                    direct_fields[mapping.harbor_field] = value

        return direct_fields, kwargs_fields

    def _build_environment_config(self) -> EnvironmentConfig:
        """Build EnvironmentConfig from config."""
        env_fields = {}
        env_kwargs = {}

        for yaml_key, mapping in ENVIRONMENT_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                if mapping.field_type == "kwargs":
                    # Pass through to environment kwargs
                    env_kwargs[mapping.harbor_field] = value
                elif mapping.harbor_field == "type":
                    # Special handling for environment type
                    if isinstance(value, str):
                        value = EnvironmentType(value)
                    env_fields[mapping.harbor_field] = value
                else:
                    env_fields[mapping.harbor_field] = value

        # Default to Daytona if neither type nor import_path specified
        if "type" not in env_fields and "import_path" not in env_fields:
            env_fields["type"] = EnvironmentType.DAYTONA

        # Add kwargs if any were collected
        if env_kwargs:
            env_fields["kwargs"] = env_kwargs

        return EnvironmentConfig(**env_fields)

    def _build_verifier_config(self) -> VerifierConfig:
        """Build VerifierConfig from config."""
        verifier_fields = {}

        for yaml_key, mapping in VERIFIER_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                verifier_fields[mapping.harbor_field] = value

        return VerifierConfig(**verifier_fields)

    def _get_trial_fields(self) -> Dict[str, Any]:
        """Get trial-level fields from config."""
        trial_fields = {}

        for yaml_key, mapping in TRIAL_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                trial_fields[mapping.harbor_field] = value

        return trial_fields

    def build_retry_config(self) -> RetryConfig:
        """
        Build RetryConfig for QueueOrchestrator from YAML config.

        Returns:
            Configured RetryConfig with exponential backoff and exception filtering.
        """
        retry_fields = {}

        for yaml_key, mapping in RETRY_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                # Handle exception sets (YAML lists -> Python sets)
                if yaml_key in ("include_exceptions", "exclude_exceptions"):
                    if isinstance(value, (list, tuple)):
                        value = set(value)
                    elif isinstance(value, str):
                        # Support comma-separated string
                        value = {s.strip() for s in value.split(",") if s.strip()}
                retry_fields[mapping.harbor_field] = value

        return RetryConfig(**retry_fields)

    def get_n_concurrent_trials(self, default: int = 16) -> int:
        """
        Get the number of concurrent trials for QueueOrchestrator.

        Args:
            default: Default concurrency if not specified in config.

        Returns:
            Number of concurrent trials to run.
        """
        mapping = ORCHESTRATOR_SCHEMA.fields.get("n_concurrent_trials")
        if mapping:
            value = self._get_field_value("n_concurrent_trials", mapping, self._cfg)
            if value is not None:
                return int(value)
        return default

    def get_reward_shaping_config(self) -> Dict[str, Any]:
        """
        Get reward shaping configuration for terminal bench generator.

        Returns:
            Dict with keys:
                - enable_reward_shaping: bool
                - reward_parser: str | None (pytest, unittest, generic, or None for auto)
                - reward_shaper: str (pass_ratio, effective_pass_ratio, weighted, etc.)
                - reward_shaping_fallback: bool
                - shaper_kwargs: dict with shaper-specific params
        """
        config = {}

        for yaml_key, mapping in REWARD_SHAPING_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                config[yaml_key] = value

        # Build shaper kwargs from shaper-specific params
        shaper_kwargs = {}

        # Threshold shaper params
        if "reward_threshold" in config:
            shaper_kwargs["threshold"] = config.pop("reward_threshold")
        if "below_threshold_scale" in config:
            shaper_kwargs["below_threshold_scale"] = config.pop("below_threshold_scale")

        # Binary partial shaper params
        if "partial_threshold" in config:
            shaper_kwargs["partial_threshold"] = config.pop("partial_threshold")
        if "partial_credit" in config:
            shaper_kwargs["partial_credit"] = config.pop("partial_credit")

        # Thinking length shaper params
        if "thinking_target_tokens" in config:
            shaper_kwargs["target_tokens"] = config.pop("thinking_target_tokens")
        if "thinking_sigma_tokens" in config:
            shaper_kwargs["sigma_tokens"] = config.pop("thinking_sigma_tokens")
        if "thinking_min_turns_ratio" in config:
            shaper_kwargs["min_thinking_turns_ratio"] = config.pop("thinking_min_turns_ratio")

        # Format quality shaper params
        if "format_required_fields" in config:
            val = config.pop("format_required_fields")
            if val is not None:
                shaper_kwargs["required_fields"] = val
        if "format_penalize_truncated" in config:
            shaper_kwargs["penalize_truncated_json"] = config.pop("format_penalize_truncated")

        # Command quality shaper params
        if "command_quality_error_penalty_weight" in config:
            shaper_kwargs["error_penalty_weight"] = config.pop("command_quality_error_penalty_weight")
        if "command_quality_min_turns" in config:
            shaper_kwargs["min_turns"] = config.pop("command_quality_min_turns")

        # Composite shaper params
        if "composite_components" in config:
            val = config.pop("composite_components")
            if val is not None:
                shaper_kwargs["components"] = val
        if "composite_verifier_shaper" in config:
            shaper_kwargs["verifier_shaper"] = config.pop("composite_verifier_shaper")

        # composite_loop shaper params. `loop_shaping` is the nested block
        # (deep-merged onto defaults inside CompositeLoopShaper); `outcome_shaper`
        # selects the outcome term's verifier shaper.
        if "loop_shaping" in config:
            val = config.pop("loop_shaping")
            if val is not None:
                # OmegaConf DictConfig -> plain dict so the shaper can mutate it.
                if OmegaConf.is_config(val):
                    val = OmegaConf.to_container(val, resolve=True)
                shaper_kwargs["loop_shaping"] = val
        if "loop_outcome_shaper" in config:
            val = config.pop("loop_outcome_shaper")
            if val is not None:
                shaper_kwargs["outcome_shaper"] = val

        # Pass trajectory shaper kwargs through for composite mode
        trajectory_shaper_kwargs = {}
        if "target_tokens" in shaper_kwargs or "sigma_tokens" in shaper_kwargs:
            trajectory_shaper_kwargs["thinking_length"] = {
                k: shaper_kwargs[k]
                for k in ["target_tokens", "sigma_tokens", "min_thinking_turns_ratio"]
                if k in shaper_kwargs
            }
        if "required_fields" in shaper_kwargs or "penalize_truncated_json" in shaper_kwargs:
            trajectory_shaper_kwargs["format_quality"] = {
                k: shaper_kwargs[k] for k in ["required_fields", "penalize_truncated_json"] if k in shaper_kwargs
            }
        if "error_penalty_weight" in shaper_kwargs or "min_turns" in shaper_kwargs:
            trajectory_shaper_kwargs["command_quality"] = {
                k: shaper_kwargs[k] for k in ["error_penalty_weight", "min_turns"] if k in shaper_kwargs
            }
        if trajectory_shaper_kwargs:
            shaper_kwargs["trajectory_shaper_kwargs"] = trajectory_shaper_kwargs

        config["shaper_kwargs"] = shaper_kwargs

        return config

    def get_log_level(self, default: str = "WARNING") -> str:
        """
        Get the log level for Harbor.

        Args:
            default: Default log level if not specified in config.

        Returns:
            Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        mapping = LOGGING_SCHEMA.fields.get("log_level")
        if mapping:
            value = self._get_field_value("log_level", mapping, self._cfg)
            if value is not None:
                return str(value).upper()
        return default

    def get_error_handling_config(self) -> Dict[str, Any]:
        """
        Get error handling configuration for RLOO-N advantage estimator.

        This controls how different exception types are treated:
        - "passthrough" exceptions: Ignore exception, use verifier reward normally
        - "mask" exceptions: Excluded from baseline (neutral - infrastructure failures)
        - "zero" exceptions: Included in baseline with reward=0 (agent failures)

        Returns:
            Dict with keys:
                - enable_error_classification: bool - whether to classify errors
                - passthrough_exceptions: Set[str] - exception names to pass through (use verifier reward)
                - mask_exceptions: Set[str] - exception names to mask (exclude from baseline)
                - zero_exceptions: Set[str] - exception names to zero (include with reward=0)
                - default_error_treatment: str - "mask", "zero", or "passthrough" for unclassified errors
        """
        config = {}

        for yaml_key, mapping in ERROR_HANDLING_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                # Convert lists to sets for faster lookup
                if yaml_key in ("passthrough_exceptions", "mask_exceptions", "zero_exceptions"):
                    if isinstance(value, (list, tuple)):
                        value = set(value)
                    elif isinstance(value, str):
                        value = {s.strip() for s in value.split(",") if s.strip()}
                config[yaml_key] = value

        return config

    def get_eval_timeout_override_sec(self, default: int = 900) -> int:
        """
        Get the timeout override for evaluation runs.

        Eval tasks may need more time than training since they don't benefit
        from retry logic the same way. Default is 900 seconds (15 minutes).

        Args:
            default: Default timeout if not specified in config.

        Returns:
            Timeout in seconds for eval runs.
        """
        mapping = EVAL_SCHEMA.fields.get("eval_timeout_override_sec")
        if mapping:
            value = self._get_field_value("eval_timeout_override_sec", mapping, self._cfg)
            if value is not None:
                return int(value)
        return default

    def get_exclude_exceptions(self) -> List[str]:
        """
        Get the list of exclude_exceptions from the retry config.

        These are the exception types that Harbor will NOT retry (i.e., they are
        considered terminal failures). Used by the generator to pre-populate
        per-step error counters so they appear as consistent time-series.

        Returns:
            List of exception type names (e.g. ["AgentTimeoutError", ...]).
        """
        mapping = RETRY_SCHEMA.fields.get("exclude_exceptions")
        if mapping:
            value = self._get_field_value("exclude_exceptions", mapping, self._cfg)
            if value is not None:
                if isinstance(value, (list, tuple)):
                    return list(value)
                elif isinstance(value, str):
                    return [s.strip() for s in value.split(",") if s.strip()]
        return []

    def get_collect_rollout_details(self, default: bool = False) -> bool:
        """
        Check if rollout details collection is enabled (for TIS in async training).

        When true, Harbor collects per-token logprobs during rollout, which are
        needed for Truncated Importance Sampling (TIS) to correct for off-policy
        bias in async training.

        Args:
            default: Default value if not specified in config.

        Returns:
            True if rollout details collection is enabled.
        """
        mapping = AGENT_SCHEMA.fields.get("collect_rollout_details")
        if mapping:
            value = self._harbor_cfg.get("collect_rollout_details", mapping.default)
            if value is not None:
                return bool(value)
        return default

    def build_trial_config(
        self,
        task_path: str,
        trials_dir: str,
        model_name: str,
        api_base: str,
        session_id: str,
        timeout_override_sec: Optional[int] = None,
    ) -> TrialConfig:
        """
        Build a complete TrialConfig for a Harbor trial.

        Args:
            task_path: Path to the task directory.
            trials_dir: Directory for trial outputs.
            model_name: Model name for Harbor (e.g., "hosted_vllm/Qwen3-8B").
            api_base: Base URL for the inference API.
            session_id: Session ID for sticky routing.
            timeout_override_sec: Optional timeout override in seconds.
                If provided, overrides the default override_timeout_sec from config.
                Useful for eval runs that may need different timeouts.

        Returns:
            Configured TrialConfig ready for Trial execution.
        """
        # Build component configs
        environment_config = self._build_environment_config()
        verifier_config = self._build_verifier_config()
        agent_direct_fields, agent_kwargs = self._build_agent_fields()
        trial_fields = self._get_trial_fields()

        # Add required agent kwargs
        agent_kwargs.update(
            {
                "api_base": api_base,
                "key": "fake_key",
                "session_id": session_id,
                "model_info": self._model_info,
            }
        )

        # Inject PRM turn_callback if configured
        if self._turn_callback is not None:
            agent_kwargs["turn_callback"] = self._turn_callback

        # Get agent name from harbor config (defaults to "terminus-2")
        # This is the Harbor AgentName value directly (e.g., "terminus-2", "oracle")
        agent_name = agent_direct_fields.pop("name", "terminus-2")

        # Apply timeout override if provided (e.g., for eval runs)
        if timeout_override_sec is not None:
            agent_direct_fields["override_timeout_sec"] = timeout_override_sec

        # Build AgentConfig
        agent_config = AgentConfig(
            name=agent_name,
            model_name=model_name,
            kwargs=agent_kwargs,
            **agent_direct_fields,
        )

        # Build TrialConfig
        return TrialConfig(
            task=TaskConfig(path=task_path),
            trials_dir=Path(trials_dir),
            environment=environment_config,
            verifier=verifier_config,
            agent=agent_config,
            **trial_fields,
        )

    @property
    def model_info(self) -> Dict[str, Any]:
        """Get the model_info dict for external use."""
        return self._model_info.copy()


# =============================================================================
# Utility functions
# =============================================================================


def get_exposed_harbor_fields() -> Dict[str, list[str]]:
    """
    Get a summary of all exposed Harbor fields for documentation.

    Returns:
        Dict mapping section names to lists of field names.
    """
    return {section_name: list(schema.get_all_field_names()) for section_name, schema in HARBOR_SCHEMA.items()}


def print_harbor_schema() -> None:
    """Print the current Harbor schema for debugging."""
    print("SkyRL Terminal Bench - Exposed Harbor Fields")
    print("=" * 50)
    for section_name, schema in HARBOR_SCHEMA.items():
        print(f"\n{section_name.upper()}:")
        for yaml_key, mapping in schema.fields.items():
            field_type = f" (kwargs)" if mapping.field_type == "kwargs" else ""
            default = f" [default: {mapping.default}]" if mapping.default is not None else ""
            print(f"  - {yaml_key} -> {mapping.harbor_field}{field_type}{default}")
