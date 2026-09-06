"""
Built-in callbacks for common training operations.

These callbacks provide default implementations for checkpointing, evaluation,
model saving, and other periodic actions that were previously inline in the
training loop.

Supports two configuration styles:
1. Legacy interval configs (ckpt_interval, eval_interval, etc.)
2. New explicit callback configs in YAML:
   ```yaml
   trainer:
     callbacks:
       - type: checkpoint
         save_steps: 10
       - type: evaluation
         eval_steps: 20
   ```
"""

import asyncio
import contextlib
import dataclasses
import os
from typing import Any, Dict, List, Optional, Type

from loguru import logger
from marinskyrl.training_completion import validate_completion_config
from omegaconf import DictConfig
import torch

from skyrl_train.config.callbacks import has_explicit_callbacks, interval_hf_export_enabled
from skyrl_train.async_rollout_state import GeneratedOutputGroup, GenerationBufferState, GenerationQueuesProvider
from skyrl_train.trajectory_runners.base import TrajectoryBatch
from skyrl_train.json_serialization import to_jsonable
from skyrl_train.utils.data_tracker import DataConsumptionState, DataConsumptionTracker
from skyrl_train.io import io
from skyrl_train.inference_engines.vllm.stats import VLLM_NUM_ENGINES_METRIC, IntervalReadMode
from skyrl_train.inference_observability import (
    InferenceMetricsSink,
    configured_inference_sinks,
    format_console_summary,
    trainer_metrics,
)

from .base import TrainerCallback, TrainerState, TrainerControl, CallbackHandler
from .types import (
    CHECKPOINT_CALLBACK_TYPE,
    HF_MODEL_SAVE_CALLBACK_TYPE,
)

# Registry mapping callback type names to classes
# This enables YAML-based callback configuration
CALLBACK_REGISTRY: Dict[str, Type[TrainerCallback]] = {}


def register_callback(name: str):
    """
    Decorator to register a callback class in the registry.

    Args:
        name: The type name to use in YAML configs (e.g., "checkpoint")

    Example:
        @register_callback("my_callback")
        class MyCallback(TrainerCallback):
            ...
    """

    def decorator(cls: Type[TrainerCallback]) -> Type[TrainerCallback]:
        CALLBACK_REGISTRY[name] = cls
        return cls

    return decorator


@register_callback(CHECKPOINT_CALLBACK_TYPE)
class CheckpointCallback(TrainerCallback):
    """
    Callback for saving training checkpoints at regular intervals.

    This replaces the inline `ckpt_interval` logic in the training loop.
    Checkpoints include model weights, optimizer state, and training state
    for resumable training.

    Args:
        save_steps: Save a checkpoint every N steps. Set to -1 or 0 to disable.
        save_on_train_end: Whether to save a final checkpoint when training ends.
    """

    def __init__(self, save_steps: int = 10, save_on_train_end: bool = True):
        self.save_steps = save_steps
        self.save_on_train_end = save_on_train_end

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.save_steps > 0 and state.global_step % self.save_steps == 0:
            control.should_save = True
        return control

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.save_on_train_end and self.save_steps > 0:
            control.should_save = True
        return control


@register_callback("evaluation")
class EvaluationCallback(TrainerCallback):
    """
    Callback for running evaluation at regular intervals.

    This replaces the inline `eval_interval` logic in the training loop.
    Evaluation runs on the validation dataset and logs metrics.

    Args:
        eval_steps: Run evaluation every N steps. Set to -1 or 0 to disable.
        eval_on_train_end: Whether to run evaluation when training ends.
        eval_before_train: Whether to run evaluation before training starts.
    """

    def __init__(
        self,
        eval_steps: int = 5,
        eval_on_train_end: bool = True,
        eval_before_train: bool = True,
    ):
        self.eval_steps = eval_steps
        self.eval_on_train_end = eval_on_train_end
        self.eval_before_train = eval_before_train

    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.eval_before_train and self.eval_steps > 0:
            control.should_evaluate = True
        return control

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.eval_steps > 0 and state.global_step % self.eval_steps == 0:
            control.should_evaluate = True
        return control

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.eval_on_train_end and self.eval_steps > 0:
            control.should_evaluate = True
        return control


@register_callback(HF_MODEL_SAVE_CALLBACK_TYPE)
class HFModelSaveCallback(TrainerCallback):
    """
    Callback for requesting Hugging Face exports at regular intervals.

    Normal training records a request beside the immutable sharded checkpoint;
    a dedicated export job later converts and optionally publishes that checkpoint.

    Args:
        save_steps: Request an HF export every N steps. Set to -1 or 0 to disable.
        save_on_train_end: Whether to request a final HF export when training ends.
    """

    def __init__(self, save_steps: int = -1, save_on_train_end: bool = True):
        self.save_steps = save_steps
        self.save_on_train_end = save_on_train_end

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.save_steps > 0 and state.global_step % self.save_steps == 0:
            control.should_save_hf_model = True
        return control

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.save_on_train_end and self.save_steps > 0:
            control.should_save_hf_model = True
        return control


@register_callback("database_registration")
class DatabaseRegistrationCallback(TrainerCallback):
    """
    Callback for registering trained models to the unified database (Supabase).

    This callback runs at training end and registers the trained model along with:
    - Training timestamps (start/end)
    - Training configuration (hyperparameters, algorithm, etc.)
    - W&B link (if available)
    - Dataset and base model references

    Requirements:
    - KEYS environment variable pointing to Supabase credentials file, OR
    - Direct env vars: SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY

    Args:
        agent_name: Name of the agent being trained (default: from terminal_bench config or "skyrl")
        enabled: Whether registration is enabled (default: True, auto-disabled if no credentials)
    """

    def __init__(
        self,
        agent_name: Optional[str] = None,
        enabled: bool = True,
    ):
        self.agent_name = agent_name
        self.enabled = enabled
        self._training_start: Optional[str] = None
        self._supabase_ready = False
        self._cfg = None

    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Record training start time and load Supabase credentials."""
        from datetime import datetime, timezone

        # Only register from rank 0
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank != 0:
            self.enabled = False
            return control

        self._training_start = datetime.now(timezone.utc).isoformat()

        # Try to load Supabase credentials
        try:
            from skyrl_train.callbacks.database import load_supabase_keys

            required_keys = ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"]
            if not all(os.environ.get(k) for k in required_keys):
                self._supabase_ready = load_supabase_keys()
            else:
                self._supabase_ready = True

            if self._supabase_ready:
                logger.info("DatabaseRegistrationCallback: Supabase credentials loaded")
            else:
                logger.warning(
                    "DatabaseRegistrationCallback: Supabase credentials not available, "
                    "model will not be registered to database"
                )
        except ImportError as e:
            logger.warning(
                f"DatabaseRegistrationCallback: database module not available ({e}), "
                "install supabase-py to enable database registration: pip install supabase"
            )
            self.enabled = False

        # Store config reference
        trainer = kwargs.get("trainer")
        if trainer is not None and hasattr(trainer, "cfg"):
            self._cfg = trainer.cfg

        return control

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Register the trained model to the database."""
        if not self.enabled or not self._supabase_ready:
            return control

        from datetime import datetime, timezone

        try:
            from skyrl_train.callbacks.database import register_trained_model
        except ImportError:
            logger.error("DatabaseRegistrationCallback: Cannot import register_trained_model")
            return control

        training_end = datetime.now(timezone.utc).isoformat()

        # Extract configuration
        cfg = self._cfg
        if cfg is None:
            logger.warning("DatabaseRegistrationCallback: No config available, skipping registration")
            return control

        # Determine agent name
        agent_name = self.agent_name
        if not agent_name:
            # Try terminal_bench config
            tb_cfg = getattr(cfg, "terminal_bench_config", None)
            if tb_cfg:
                harbor = getattr(tb_cfg, "harbor", None)
                if harbor:
                    agent_name = getattr(harbor, "name", None)
            if not agent_name:
                agent_name = os.environ.get("TRAINING_AGENT_NAME", "skyrl")

        # Get model path and dataset info
        policy_cfg = getattr(cfg.trainer, "policy", None)
        base_model_name = None
        if policy_cfg:
            model_cfg = getattr(policy_cfg, "model", None)
            if model_cfg:
                base_model_name = getattr(model_cfg, "path", None)

        # Get dataset names
        data_cfg = getattr(cfg, "data", None)
        train_data = getattr(data_cfg, "train_data", []) if data_cfg else []
        dataset_names = list(train_data) if isinstance(train_data, (list, tuple)) else [train_data]

        # Get HF repo ID for weights location
        hf_hub_repo_id = getattr(cfg.trainer, "hf_hub_repo_id", None)

        # Get W&B link
        wandb_link = None
        try:
            import wandb

            if wandb.run is not None:
                wandb_link = wandb.run.url
        except Exception:
            pass

        training_params = {
            "trainer": to_jsonable(cfg.trainer) if hasattr(cfg, "trainer") else {},
            "generator": to_jsonable(cfg.generator) if hasattr(cfg, "generator") else {},
            "algorithm": str(getattr(cfg.trainer.algorithm, "advantage_estimator", "unknown")),
        }

        # Build registration record
        record = {
            "agent_name": agent_name,
            "training_start": self._training_start,
            "training_end": training_end,
            "created_by": os.environ.get("JOB_CREATOR", ""),
            "base_model_name": base_model_name,
            "dataset_names": dataset_names,
            "training_type": "RL",
            "training_parameters": training_params,
            "wandb_link": wandb_link or "",
            "traces_location_s3": os.environ.get("TRACE_S3_PATH", ""),
            "model_name": hf_hub_repo_id,
        }

        logger.info(
            f"DatabaseRegistrationCallback: Registering model to database "
            f"(agent={agent_name}, base_model={base_model_name}, datasets={dataset_names})"
        )

        try:
            result = register_trained_model(record)

            if result.get("success"):
                model = result.get("model", {})
                model_name = model.get("name", "unknown")
                if result.get("exists"):
                    logger.info(f"DatabaseRegistrationCallback: Model '{model_name}' already exists in database")
                elif result.get("updated"):
                    logger.info(f"DatabaseRegistrationCallback: Updated existing model '{model_name}'")
                else:
                    logger.info(f"DatabaseRegistrationCallback: Registered new model '{model_name}'")
            else:
                logger.error(f"DatabaseRegistrationCallback: Registration failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"DatabaseRegistrationCallback: Exception during registration: {e}")

        return control


@register_callback("ref_model_update")
class RefModelUpdateCallback(TrainerCallback):
    """
    Callback for updating the reference model with policy weights at epoch boundaries.

    This replaces the inline `update_ref_every_epoch` logic in the training loop.
    The reference model is used for KL divergence calculations in algorithms
    like PPO and GRPO.

    Args:
        update_every_epoch: Whether to update the reference model at the end of each epoch.
    """

    def __init__(self, update_every_epoch: bool = False):
        self.update_every_epoch = update_every_epoch
        self._should_update_ref = False

    def on_epoch_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        # Mark that we should update ref model
        # The actual update is handled by the trainer when it processes this flag
        if self.update_every_epoch and not state.is_last_step:
            # Skip updating ref at the end of the last epoch (as the original code did)
            self._should_update_ref = True
        return control

    @property
    def should_update_ref(self) -> bool:
        """Check if ref model should be updated and reset the flag."""
        result = self._should_update_ref
        self._should_update_ref = False
        return result


@register_callback("progress")
class ProgressCallback(TrainerCallback):
    """
    Callback for tracking and displaying training progress.

    This provides a central place for progress tracking without modifying
    the core training loop.

    Args:
        log_interval: Log progress every N steps. Default is every step.
    """

    def __init__(self, log_interval: int = 1):
        self.log_interval = log_interval

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.log_interval > 0 and state.global_step % self.log_interval == 0:
            logger.info(
                f"Step {state.global_step}/{state.total_steps} (Epoch {state.epoch + 1}, Step {state.step_in_epoch})"
            )
        return control


@register_callback("logging")
class LoggingCallback(TrainerCallback):
    """
    Callback for logging metrics to tracking systems (WandB, MLflow).

    This callback handles the actual logging to external tracking systems.
    It's always enabled by default.

    Args:
        log_every_step: Whether to log after every step. Default True.
    """

    def __init__(self, log_every_step: bool = True):
        self.log_every_step = log_every_step

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.log_every_step:
            control.should_log = True
        return control


class PreflightGateError(Exception):
    """Raised when the pre-flight reward gate fails with on_failure='abort'."""


@register_callback("preflight_gate")
class PreflightGateCallback(TrainerCallback):
    """Pre-flight reward gate: abort training when the reward distribution is
    outside the band where RLOO has usable within-group variance.

    DEFAULT-OFF. When enabled, checks mean per-sample reward after the first
    training step's rollouts are scored (before the second step).  On failure
    with ``on_failure="abort"`` raises ``PreflightGateError``; with
    ``on_failure="warn"`` logs and continues.
    """

    def __init__(
        self,
        enabled: bool = False,
        min_reward: float = 0.25,
        max_reward: float = 0.75,
        on_failure: str = "abort",
        num_trials: int = 256,
    ):
        self.enabled = enabled
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.on_failure = on_failure
        self.num_trials = num_trials
        self._checked = False

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self._checked or not self.enabled:
            return control
        self._checked = True

        trainer = kwargs.get("trainer")
        if trainer is None:
            return control

        rewards = self._extract_step_rewards(trainer)
        if not rewards:
            # An enabled gate that quietly does nothing is the failure this whole
            # mechanism exists to prevent, so say so at error level rather than
            # letting the run proceed looking gated.
            logger.error(
                "[preflight] Gate is ENABLED but step 1 exposed no per-sample rewards, "
                "so nothing was checked and this run is UNGATED. Expected the trainer to "
                "set _current_step_rewards during reward post-processing."
            )
            return control

        if len(rewards) < self.num_trials:
            # Not a failure: the sample count is set by train_batch_size x
            # n_samples_per_prompt, not by the dataset. Report it so a verdict drawn
            # from a thin sample is not read as a firm one.
            logger.warning(
                f"[preflight] Checking {len(rewards)} samples, fewer than the requested "
                f"num_trials={self.num_trials}; the estimate is correspondingly noisier."
            )

        from skyrl_train.utils.preflight_gate import check_preflight_gate

        result = check_preflight_gate(rewards, self.min_reward, self.max_reward)
        if not result.passed and self.on_failure == "abort":
            raise PreflightGateError(result.message)

        return control

    @staticmethod
    def _extract_step_rewards(trainer) -> List[float]:
        """Extract per-sample scalar rewards from the trainer's current batch."""
        for attr in ("_current_step_rewards", "step_rewards"):
            val = getattr(trainer, attr, None)
            if val is not None:
                return [float(r) for r in val]
        return []


@register_callback("inference_stats")
class InferenceStatsCallback(TrainerCallback):
    """Collect one canonical inference snapshot and fan it out to independent sinks."""

    def __init__(
        self,
        log_every_steps: int = 1,
        log_to_console: bool = True,
        log_to_tracker: bool = True,
        console_log_level: str = "info",
        poll_interval_seconds: float = 5.0,
        sinks: tuple[InferenceMetricsSink, ...] | None = None,
    ):
        self.log_every_steps = log_every_steps
        self.log_to_console = log_to_console
        self.log_to_tracker = log_to_tracker
        self.console_log_level = console_log_level.lower()
        self.poll_interval_seconds = poll_interval_seconds
        self._sinks = sinks
        self._inference_engine_client = None
        self._poll_task: asyncio.Task | None = None
        self._read_lock = asyncio.Lock()
        self._step = 0

    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        trainer = kwargs.get("trainer")
        if trainer is not None:
            self._inference_engine_client = getattr(trainer, "inference_engine_client", None)
            if self._inference_engine_client is None:
                logger.warning(
                    "InferenceStatsCallback: No inference_engine_client found on trainer. Stats collection will be disabled."
                )
        if self._sinks is None:
            self._sinks = configured_inference_sinks()
        return control

    async def on_train_begin_async(self, state: TrainerState, control: TrainerControl, **kwargs):
        self.on_train_begin(state, control, **kwargs)
        self._step = state.global_step
        if self._inference_engine_client is not None and self._sinks:
            try:
                async with self._read_lock:
                    snapshot = await self._inference_engine_client.get_stats(read_mode=IntervalReadMode.PEEK)
                self._publish_sinks(snapshot, self._step)
            except Exception:
                logger.warning("InferenceStatsCallback: initial collection failed", exc_info=True)
            if self.poll_interval_seconds > 0:
                self._poll_task = asyncio.create_task(self._poll(), name="vllm-stats-poll")
        return control

    async def on_train_end_async(self, state: TrainerState, control: TrainerControl, **kwargs):
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None
        return control

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval_seconds)
            try:
                async with self._read_lock:
                    snapshot = await self._inference_engine_client.get_stats(read_mode=IntervalReadMode.PEEK)
                self._publish_sinks(snapshot, self._step)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("InferenceStatsCallback: periodic collection failed", exc_info=True)

    async def on_step_end_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.log_every_steps <= 0 or state.global_step % self.log_every_steps != 0:
            return control
        if self._inference_engine_client is None:
            return control

        self._step = state.global_step
        try:
            async with self._read_lock:
                snapshot = await self._inference_engine_client.get_stats(read_mode=IntervalReadMode.RESET)
        except Exception:
            logger.warning("InferenceStatsCallback: Failed to collect stats", exc_info=True)
            return control

        metrics = trainer_metrics(snapshot)
        if not metrics:
            return control
        if self.log_to_tracker:
            kwargs["trainer"].all_metrics.update(metrics)
        if self.log_to_console and VLLM_NUM_ENGINES_METRIC in metrics:
            log = logger.debug if self.console_log_level == "debug" else logger.info
            log(format_console_summary(metrics, state.global_step))
        self._publish_sinks(snapshot, state.global_step)
        return control

    def _publish_sinks(self, snapshot, step: int) -> None:
        for sink in self._sinks or ():
            try:
                sink.publish(snapshot, step)
            except Exception:
                logger.warning(f"InferenceStatsCallback: {type(sink).__name__} failed", exc_info=True)


def create_default_callbacks(cfg: DictConfig) -> List[TrainerCallback]:
    """
    Create the default set of callbacks based on trainer configuration.

    Supports two configuration styles:

    1. **New style** (explicit callbacks list):
       ```yaml
       trainer:
         callbacks:
           - type: checkpoint
             save_steps: 10
           - type: evaluation
             eval_steps: 20
       ```

    2. **Legacy style** (interval configs):
       ```yaml
       trainer:
         ckpt_interval: 10
         eval_interval: 20
       ```

    If explicit 'callbacks' config is present, it takes precedence.
    Otherwise, callbacks are created from legacy interval configs.

    Args:
        cfg: Training configuration (OmegaConf DictConfig)

    Returns:
        List of configured callbacks
    """
    validate_completion_config(cfg)
    # Check for new-style explicit callback configuration
    if has_explicit_callbacks(cfg):
        logger.info("Using explicit callback configuration from YAML")
        callbacks = create_callbacks_from_config(cfg)
        # Always add logging callback if not explicitly configured
        has_logging = any(isinstance(cb, LoggingCallback) for cb in callbacks)
        if not has_logging:
            callbacks.append(LoggingCallback())
        return callbacks

    # Fall back to legacy interval-based configuration
    logger.debug("Using legacy interval-based callback configuration")
    callbacks = []

    # Checkpoint callback
    ckpt_interval = getattr(cfg.trainer, "ckpt_interval", 10)
    if ckpt_interval > 0:
        callbacks.append(CheckpointCallback(save_steps=ckpt_interval))

    # Evaluation callback
    eval_interval = getattr(cfg.trainer, "eval_interval", 5)
    eval_before_train = getattr(cfg.trainer, "eval_before_train", True)
    if eval_interval > 0:
        callbacks.append(
            EvaluationCallback(
                eval_steps=eval_interval,
                eval_before_train=eval_before_train,
            )
        )

    # HF model save callback
    if interval_hf_export_enabled(cfg):
        callbacks.append(HFModelSaveCallback(save_steps=int(cfg.trainer.hf_save_interval)))

    # Reference model update callback
    update_ref_every_epoch = getattr(cfg.trainer, "update_ref_every_epoch", False)
    if update_ref_every_epoch:
        callbacks.append(RefModelUpdateCallback(update_every_epoch=True))

    # Database registration callback (auto-enabled, gracefully disabled if no credentials)
    enable_db_registration = getattr(cfg.trainer, "enable_db_registration", True)
    if enable_db_registration:
        # Get agent name from terminal_bench config if available
        agent_name = None
        tb_cfg = getattr(cfg, "terminal_bench_config", None)
        if tb_cfg:
            harbor = getattr(tb_cfg, "harbor", None)
            if harbor:
                agent_name = getattr(harbor, "name", None)
        callbacks.append(DatabaseRegistrationCallback(agent_name=agent_name))

    # Pre-flight reward gate (default-off)
    gate_cfg = getattr(cfg.trainer, "preflight_gate", None)
    gate_enabled = getattr(gate_cfg, "enabled", False) if gate_cfg else False
    if gate_enabled:
        callbacks.append(
            PreflightGateCallback(
                enabled=True,
                min_reward=getattr(gate_cfg, "min_reward", 0.25),
                max_reward=getattr(gate_cfg, "max_reward", 0.75),
                on_failure=getattr(gate_cfg, "on_failure", "abort"),
                num_trials=getattr(gate_cfg, "num_trials", 256),
            )
        )

    # Inference stats callback (enabled when using the vLLM backend). This combines
    # engine and HTTP bridge stats without relying on Ray log-to-driver forwarding.
    generator_backend = getattr(cfg.generator, "backend", None)
    inference_stats_interval = getattr(cfg.generator, "inference_stats_interval", 1)
    if generator_backend == "vllm" and inference_stats_interval > 0:
        callbacks.append(
            InferenceStatsCallback(
                log_every_steps=inference_stats_interval,
                log_to_console=True,
                log_to_tracker=True,
            )
        )

    # Logging callback (always enabled)
    callbacks.append(LoggingCallback())

    return callbacks


class DefaultCallbackHandler(CallbackHandler):
    """
    A callback handler that initializes with default callbacks based on config.

    This provides backward compatibility by recreating the original inline
    behavior through callbacks.

    Example:
        ```python
        handler = DefaultCallbackHandler(cfg)
        # Adds all default callbacks based on config intervals
        ```
    """

    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        """
        Initialize with default callbacks from config, plus any custom callbacks.

        Args:
            cfg: Training configuration. If provided, creates default callbacks.
            callbacks: Additional custom callbacks to add after defaults.
        """
        default_callbacks = []
        if cfg is not None:
            default_callbacks = create_default_callbacks(cfg)

        all_callbacks = default_callbacks + (callbacks or [])
        super().__init__(all_callbacks)

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        additional_callbacks: Optional[List[TrainerCallback]] = None,
    ) -> "DefaultCallbackHandler":
        """
        Create a handler from config with optional additional callbacks.

        Args:
            cfg: Training configuration
            additional_callbacks: Custom callbacks to add after defaults

        Returns:
            Configured callback handler
        """
        return cls(cfg=cfg, callbacks=additional_callbacks)


def create_callback_from_config(callback_config: Dict[str, Any]) -> TrainerCallback:
    """
    Create a callback instance from a YAML config dictionary.

    Args:
        callback_config: Dictionary with 'type' key and callback-specific params.
            Example: {"type": "checkpoint", "save_steps": 10}

    Returns:
        Instantiated callback

    Raises:
        ValueError: If callback type is unknown or missing
    """
    if "type" not in callback_config:
        raise ValueError(f"Callback config missing 'type' key: {callback_config}")

    callback_type = callback_config["type"]
    if callback_type not in CALLBACK_REGISTRY:
        available = ", ".join(CALLBACK_REGISTRY.keys())
        raise ValueError(f"Unknown callback type '{callback_type}'. Available types: {available}")

    # Get the callback class and instantiate with remaining params
    callback_cls = CALLBACK_REGISTRY[callback_type]
    params = {k: v for k, v in callback_config.items() if k != "type"}

    try:
        return callback_cls(**params)
    except TypeError as e:
        raise ValueError(f"Invalid parameters for callback '{callback_type}': {e}") from e


def create_callbacks_from_config(cfg: DictConfig) -> List[TrainerCallback]:
    """
    Create callbacks from explicit YAML configuration.

    This supports the new-style callback configuration:
    ```yaml
    trainer:
      callbacks:
        - type: checkpoint
          save_steps: 10
        - type: evaluation
          eval_steps: 20
          eval_before_train: false
    ```

    Args:
        cfg: Training configuration with optional 'callbacks' list

    Returns:
        List of instantiated callbacks (empty if no callbacks configured)
    """
    callbacks_config = getattr(cfg.trainer, "callbacks", None)
    if callbacks_config is None:
        return []

    callbacks = []
    for callback_config in callbacks_config:
        # Convert OmegaConf to dict if needed
        if hasattr(callback_config, "items"):
            config_dict = dict(callback_config)
        else:
            config_dict = callback_config

        try:
            callback = create_callback_from_config(config_dict)
            callbacks.append(callback)
            logger.debug(f"Created callback: {callback.__class__.__name__}")
        except ValueError as e:
            logger.error(f"Failed to create callback: {e}")
            raise

    return callbacks


def get_available_callback_types() -> List[str]:
    """Get list of available callback type names for YAML configs."""
    return list(CALLBACK_REGISTRY.keys())


@register_callback("data_tracking")
class DataTrackingCallback(TrainerCallback):
    """
    Persists data consumption state as a checkpoint artifact via the callback system.

    This replaces the inline fully_async_state.pt writing/loading that was previously
    embedded in the fully async trainer. By using the callback system:
    - Epoch-end UID clearing happens AFTER checkpoint saves (no more race condition)
    - Data state persistence is decoupled from trainer implementation
    - Backward compatible with legacy fully_async_state.pt checkpoints

    Hooks used:
    - on_save: writes data_consumption_state.pt to the checkpoint directory
    - on_epoch_end_async: clears epoch-scoped UIDs via tracker.on_epoch_end()
    """

    error_behavior = "raise"  # data tracking errors should stop training
    ARTIFACT_NAME = "data_consumption_state.pt"

    def __init__(self, tracker: DataConsumptionTracker):
        assert isinstance(tracker, DataConsumptionTracker)
        self._tracker = tracker

    def on_save(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        trainer = kwargs.get("trainer")
        if trainer is None:
            logger.warning("DataTrackingCallback.on_save: no trainer in kwargs, skipping")
            return control

        ckpt_path = os.path.join(
            trainer.cfg.trainer.ckpt_path,
            f"global_step_{state.global_step}",
        )
        data_state = self._tracker.get_state()
        data_state.global_step = state.global_step
        artifact_path = os.path.join(ckpt_path, self.ARTIFACT_NAME)
        with io.open_file(artifact_path, "wb") as f:
            torch.save(dataclasses.asdict(data_state), f)
        logger.info(
            f"Saved data consumption state to {artifact_path} "
            f"(epoch={data_state.epoch}, consumed_in_epoch={len(data_state.consumed_uids_in_epoch)}, "
            f"total={data_state.total_samples_consumed})"
        )
        return control

    async def on_epoch_end_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        await self._tracker.on_epoch_end()
        return control

    @staticmethod
    def load_from_checkpoint(
        ckpt_path: str,
        tracker: DataConsumptionTracker,
    ) -> bool:
        """Load data consumption state from a checkpoint directory.

        Tries the new data_consumption_state.pt first, then falls back to
        legacy fully_async_state.pt for backward compatibility.

        Returns True if state was loaded, False if no artifact found.
        """

        # Try new format first
        artifact_path = os.path.join(ckpt_path, DataTrackingCallback.ARTIFACT_NAME)
        if io.exists(artifact_path):
            with io.open_file(artifact_path, "rb") as f:
                raw = torch.load(f, map_location="cpu", weights_only=False)
            state = DataConsumptionState(**raw)
            tracker.load_state(state)
            return True

        # Fall back to legacy fully_async_state.pt
        legacy_path = os.path.join(ckpt_path, "fully_async_state.pt")
        if io.exists(legacy_path):
            with io.open_file(legacy_path, "rb") as f:
                legacy = torch.load(f, map_location="cpu", weights_only=False)
            if "consumed_uids" in legacy:
                consumed = legacy["consumed_uids"]
                # Reconstruct a DataConsumptionState from legacy format.
                # We don't know the exact epoch or total, so estimate from global_step.
                # Extract global_step from the checkpoint directory name.
                dir_name = os.path.basename(ckpt_path)
                global_step = int(dir_name.split("_")[-1]) if "global_step_" in dir_name else 0
                state = DataConsumptionState(
                    global_step=global_step,
                    epoch=global_step // tracker._num_steps_per_epoch,
                    consumed_uids_in_epoch=list(consumed),
                    total_samples_consumed=len(consumed)
                    + (global_step // tracker._num_steps_per_epoch)
                    * tracker._num_steps_per_epoch
                    * tracker._mini_batch_size,
                )
                tracker.load_state(state)
                logger.info(f"Loaded legacy fully_async_state.pt with {len(consumed)} consumed UIDs")
                return True

        return False


class BufferCheckpointCallback(TrainerCallback):
    """Persist async rollout work with each checkpoint and during shutdown.

    Saves completed and admitted output groups plus stale-group retry prompts so
    resume preserves every dataset row still needed by the current epoch.
    """

    ARTIFACT_NAME = "generation_buffer_state.pt"
    error_behavior = "raise"

    def __init__(self) -> None:
        self._queues: Optional[GenerationQueuesProvider] = None

    def bind_queues(self, queues: GenerationQueuesProvider) -> None:
        """Select the current epoch's queues for checkpoint persistence."""
        self._queues = queues

    def has_bound_queues(self) -> bool:
        """Return whether the current epoch has exposed its generation queues."""
        return self._queues is not None

    def has_shutdown_state(self) -> bool:
        """Return whether the bound queues contain work needed after shutdown."""
        if self._queues is None:
            return False
        state = self._queues.shutdown_snapshot()
        return bool(state.completed_groups or state.admitted_groups or state.retry_prompts)

    @staticmethod
    def _serialize_groups(groups: List[GeneratedOutputGroup]) -> List[dict]:
        return [
            {
                "trajectory_batch": dict(item.trajectory_batch),
                "uid": item.uid,
                "earliest_model_step": item.earliest_model_step,
                "source_prompts": item.source_prompts,
            }
            for item in groups
        ]

    async def _save_bound_state(
        self,
        checkpoint_path: str,
        buffer_state: GenerationBufferState,
    ) -> None:
        completed = self._serialize_groups(buffer_state.completed_groups)
        admitted = self._serialize_groups(buffer_state.admitted_groups)
        retry_prompts = buffer_state.retry_prompts

        artifact_path = os.path.join(checkpoint_path, self.ARTIFACT_NAME)

        def save_state() -> None:
            with io.open_file(artifact_path, "wb") as f:
                torch.save(
                    {
                        "completed_groups": completed,
                        "admitted_groups": admitted,
                        "retry_prompts": retry_prompts,
                    },
                    f,
                )

        await asyncio.to_thread(save_state)
        logger.info(
            "Saved {} completed, {} admitted generation groups, and {} pending retries to {}",
            len(completed),
            len(admitted),
            len(retry_prompts),
            artifact_path,
        )

    async def flush_to_checkpoint(self, checkpoint_path: str) -> None:
        """Persist all resumable work, including a trained but uncheckpointed batch."""
        if self._queues is None:
            raise RuntimeError("BufferCheckpointCallback queues were not bound before shutdown flush")
        await self._save_bound_state(checkpoint_path, self._queues.shutdown_snapshot())

    async def on_save_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        trainer = kwargs.get("trainer")
        if trainer is None:
            raise RuntimeError("BufferCheckpointCallback requires trainer context during checkpoint save")
        if self._queues is None:
            raise RuntimeError("BufferCheckpointCallback queues were not bound before checkpoint save")

        buffer_state = self._queues.snapshot()
        if not (buffer_state.completed_groups or buffer_state.admitted_groups or buffer_state.retry_prompts):
            return control

        ckpt_path = os.path.join(
            trainer.cfg.trainer.ckpt_path,
            f"global_step_{state.global_step}",
        )
        await self._save_bound_state(ckpt_path, buffer_state)

        return control

    @staticmethod
    def load_buffer_state(ckpt_path: str) -> GenerationBufferState:
        """Load completed, admitted, and retryable rollout work from a checkpoint."""

        artifact_path = os.path.join(ckpt_path, BufferCheckpointCallback.ARTIFACT_NAME)
        if not io.exists(artifact_path):
            return GenerationBufferState(completed_groups=[], retry_prompts=[])

        with io.open_file(artifact_path, "rb") as f:
            state = torch.load(f, map_location="cpu", weights_only=False)

        def deserialize_groups(entries: List[dict]) -> List[GeneratedOutputGroup]:
            groups = []
            for entry in entries:
                trajectory_batch: TrajectoryBatch = entry["trajectory_batch"]
                groups.append(
                    GeneratedOutputGroup(
                        trajectory_batch=trajectory_batch,
                        uid=entry["uid"],
                        earliest_model_step=entry["earliest_model_step"],
                        source_prompts=entry["source_prompts"],
                    )
                )
            return groups

        items = deserialize_groups(state["completed_groups"])
        admitted = deserialize_groups(state.get("admitted_groups", []))
        return GenerationBufferState(
            completed_groups=items,
            retry_prompts=state["retry_prompts"],
            admitted_groups=admitted,
        )
