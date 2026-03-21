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

from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from loguru import logger

from .base import TrainerCallback, TrainerState, TrainerControl, CallbackHandler

if TYPE_CHECKING:
    from omegaconf import DictConfig


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


@register_callback("checkpoint")
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


@register_callback("hf_model_save")
class HFModelSaveCallback(TrainerCallback):
    """
    Callback for saving models in HuggingFace format at regular intervals.

    This replaces the inline `hf_save_interval` logic in the training loop.
    HF format models can be loaded directly with transformers and pushed to
    the HuggingFace Hub.

    Args:
        save_steps: Save HF model every N steps. Set to -1 or 0 to disable.
        save_on_train_end: Whether to save final HF model when training ends.
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


@register_callback("hf_hub_upload")
class HFHubUploadCallback(TrainerCallback):
    """
    Callback for uploading HuggingFace format models to HuggingFace Hub.

    This callback uploads models saved by HFModelSaveCallback to a HuggingFace Hub
    repository. It runs asynchronously after the HF model save to avoid blocking
    training.

    The callback requires:
    - HF_TOKEN environment variable or huggingface-cli login
    - huggingface_hub package installed

    Args:
        repo_id: HuggingFace Hub repository ID (e.g., "username/model-name").
            If None, uses HF_HUB_REPO_ID environment variable.
        upload_steps: Upload every N steps. Should match hf_save_interval.
            Set to -1 or 0 to disable periodic uploads.
        upload_on_train_end: Whether to upload the final model when training ends.
        private: Whether to create a private repository.
        revision: Branch to upload to (default: "main").
        path_in_repo_prefix: Prefix for upload path (default: "checkpoints").
            Models are uploaded to "{prefix}/step_{N}/".
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        upload_steps: int = -1,
        upload_on_train_end: bool = True,
        private: bool = False,
        revision: str = "main",
        path_in_repo_prefix: str = "checkpoints",
    ):
        import os

        self.repo_id = repo_id or os.environ.get("HF_HUB_REPO_ID")
        self.upload_steps = upload_steps
        self.upload_on_train_end = upload_on_train_end
        self.private = private
        self.revision = revision
        self.path_in_repo_prefix = path_in_repo_prefix
        self._pending_uploads: List[int] = []  # Steps that need uploading
        self._export_path: Optional[str] = None
        self._api = None

    def _get_api(self):
        """Lazy-load HuggingFace Hub API."""
        if self._api is None:
            try:
                from huggingface_hub import HfApi
                self._api = HfApi()
            except ImportError:
                logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
                raise
        return self._api

    def _ensure_repo_exists(self) -> bool:
        """Ensure the HuggingFace Hub repository exists, creating if needed."""
        if not self.repo_id:
            logger.warning("HFHubUploadCallback: No repo_id configured, skipping upload")
            return False

        try:
            api = self._get_api()
            api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=self.private,
                exist_ok=True,
            )
            return True
        except Exception as e:
            logger.error(f"HFHubUploadCallback: Failed to create/access repo {self.repo_id}: {e}")
            return False

    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Store export_path from trainer config at training start."""
        trainer = kwargs.get("trainer")
        if trainer is not None and hasattr(trainer, "cfg"):
            self._export_path = getattr(trainer.cfg.trainer, "export_path", None)
            logger.info(
                f"HFHubUploadCallback initialized: repo={self.repo_id}, "
                f"upload_steps={self.upload_steps}, export_path={self._export_path}"
            )
        return control

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Queue upload after HF model save steps."""
        if self.upload_steps > 0 and state.global_step % self.upload_steps == 0:
            self._pending_uploads.append(state.global_step)
        return control

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Upload final model and process any pending uploads."""
        if self.upload_on_train_end and self.upload_steps > 0:
            # Add final step if not already pending
            if state.global_step not in self._pending_uploads:
                self._pending_uploads.append(state.global_step)

        # Process all pending uploads
        self._process_pending_uploads()
        return control

    def _process_pending_uploads(self) -> None:
        """Process all pending uploads."""
        if not self._pending_uploads:
            return

        if not self._export_path:
            logger.warning("HFHubUploadCallback: No export_path configured, skipping uploads")
            return

        if not self._ensure_repo_exists():
            return

        import os
        from pathlib import Path

        api = self._get_api()

        for step in self._pending_uploads:
            model_path = Path(self._export_path) / f"global_step_{step}" / "policy"

            if not model_path.exists():
                logger.warning(f"HFHubUploadCallback: Model path not found: {model_path}")
                continue

            path_in_repo = f"{self.path_in_repo_prefix}/step_{step}"

            try:
                logger.info(f"HFHubUploadCallback: Uploading {model_path} to {self.repo_id}/{path_in_repo}")
                api.upload_folder(
                    folder_path=str(model_path),
                    repo_id=self.repo_id,
                    path_in_repo=path_in_repo,
                    repo_type="model",
                    revision=self.revision,
                    commit_message=f"Upload checkpoint at step {step}",
                )
                logger.info(f"HFHubUploadCallback: Successfully uploaded step {step}")
            except Exception as e:
                logger.error(f"HFHubUploadCallback: Failed to upload step {step}: {e}")

        self._pending_uploads.clear()

    async def on_train_end_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Async version - uploads in background to not block training end."""
        import asyncio

        if self.upload_on_train_end and self.upload_steps > 0:
            if state.global_step not in self._pending_uploads:
                self._pending_uploads.append(state.global_step)

        # Run uploads in thread pool to not block
        if self._pending_uploads:
            await asyncio.to_thread(self._process_pending_uploads)

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
        import os
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

        import os
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

        # Build training parameters (serialize config)
        def _to_jsonable(obj):
            """Convert OmegaConf to JSON-serializable dict."""
            if hasattr(obj, "items"):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        training_params = {
            "trainer": _to_jsonable(cfg.trainer) if hasattr(cfg, "trainer") else {},
            "generator": _to_jsonable(cfg.generator) if hasattr(cfg, "generator") else {},
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
                f"Step {state.global_step}/{state.total_steps} "
                f"(Epoch {state.epoch + 1}, Step {state.step_in_epoch})"
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


@register_callback("vllm_stats")
class VLLMStatsCallback(TrainerCallback):
    """
    Callback for collecting and logging vLLM inference engine statistics.

    This callback queries vLLM engines directly for their stats (prompt/generation
    throughput, KV cache usage, request counts) bypassing Ray's log-to-driver
    functionality which can be unreliable.

    Stats are logged to both console (loguru) and wandb (if available).

    Args:
        log_every_steps: Log stats every N steps. Default 1 (every step).
        log_to_console: Whether to log stats to console via loguru. Default True.
        log_to_wandb: Whether to log stats to wandb. Default True.
        console_log_level: Log level for console output ("info", "debug"). Default "info".
    """

    def __init__(
        self,
        log_every_steps: int = 1,
        log_to_console: bool = True,
        log_to_wandb: bool = True,
        console_log_level: str = "info",
    ):
        self.log_every_steps = log_every_steps
        self.log_to_console = log_to_console
        self.log_to_wandb = log_to_wandb
        self.console_log_level = console_log_level.lower()
        self._wandb_available: Optional[bool] = None
        self._inference_engine_client = None

    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Cache reference to inference engine client at training start."""
        trainer = kwargs.get("trainer")
        if trainer is not None:
            self._inference_engine_client = getattr(trainer, "inference_engine_client", None)
            if self._inference_engine_client is None:
                logger.warning(
                    "VLLMStatsCallback: No inference_engine_client found on trainer. "
                    "Stats collection will be disabled."
                )
        return control

    async def on_step_end_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Query engines for stats and log them."""
        if self.log_every_steps <= 0:
            return control

        if state.global_step % self.log_every_steps != 0:
            return control

        if self._inference_engine_client is None:
            return control

        try:
            stats = await self._inference_engine_client.get_stats()
            self._log_stats(stats, state.global_step)
        except Exception as e:
            logger.warning(f"VLLMStatsCallback: Failed to collect stats: {e}")

        return control

    def _log_stats(self, stats: Dict[str, Any], global_step: int) -> None:
        """Log stats to console and wandb."""
        num_engines = stats.get("num_engines", 0)
        if num_engines == 0:
            return

        # Build log message with both peak and median stats
        total_samples = stats.get("total_samples", 0)
        total_active = stats.get("total_active_samples", 0)

        # Use new field names if available, fall back to legacy
        peak_running = stats.get("total_peak_running_reqs", stats.get("total_running_reqs", 0))
        peak_waiting = stats.get("total_peak_waiting_reqs", stats.get("total_waiting_reqs", 0))
        peak_prompt_tp = stats.get("avg_peak_prompt_throughput", stats.get("avg_prompt_throughput", 0.0))
        peak_gen_tp = stats.get("avg_peak_generation_throughput", stats.get("avg_generation_throughput", 0.0))
        peak_kv_cache = stats.get("avg_peak_gpu_cache_usage_perc", stats.get("avg_gpu_cache_usage_perc", 0.0))

        median_running = stats.get("avg_median_running_reqs", 0.0)
        median_waiting = stats.get("avg_median_waiting_reqs", 0.0)
        median_prompt_tp = stats.get("avg_median_prompt_throughput", 0.0)
        median_gen_tp = stats.get("avg_median_generation_throughput", 0.0)
        median_kv_cache = stats.get("avg_median_gpu_cache_usage_perc", 0.0)

        # Latency stats
        prefill_mean = stats.get("avg_latency_prefill_mean", 0.0)
        prefill_p90 = stats.get("max_latency_prefill_p90", 0.0)
        decode_mean = stats.get("avg_latency_decode_mean", 0.0)
        decode_p90 = stats.get("max_latency_decode_p90", 0.0)
        e2e_mean = stats.get("avg_latency_e2e_mean", 0.0)
        e2e_p90 = stats.get("max_latency_e2e_p90", 0.0)
        queued_mean = stats.get("avg_latency_queued_mean", 0.0)
        ttft_mean = stats.get("avg_latency_ttft_mean", 0.0)
        total_finished = stats.get("total_finished_requests", 0)
        total_preempted = stats.get("total_preempted_reqs", 0)

        msg = (
            f"vLLM Stats (step {global_step}): "
            f"engines={num_engines}, "
            f"running(peak/med)={peak_running}/{median_running:.0f}, "
            f"waiting(peak/med)={peak_waiting}/{median_waiting:.0f}, "
            f"prompt_tp(peak/med)={peak_prompt_tp:.1f}/{median_prompt_tp:.1f} tok/s, "
            f"gen_tp(peak/med)={peak_gen_tp:.1f}/{median_gen_tp:.1f} tok/s, "
            f"kv_cache(peak/med)={peak_kv_cache:.1f}/{median_kv_cache:.1f}%"
        )
        if total_finished > 0:
            msg += (
                f", prefill(mean/p90)={prefill_mean:.2f}/{prefill_p90:.2f}s"
                f", decode(mean/p90)={decode_mean:.2f}/{decode_p90:.2f}s"
                f", e2e(mean/p90)={e2e_mean:.2f}/{e2e_p90:.2f}s"
                f", queued={queued_mean:.2f}s, ttft={ttft_mean:.2f}s"
                f", finished={total_finished}, preempted={total_preempted}"
            )
        if total_samples > 0:
            msg += f", samples={total_active}/{total_samples}"

        # Log to console
        if self.log_to_console:
            if self.console_log_level == "debug":
                logger.debug(msg)
            else:
                logger.info(msg)

        # Log to wandb
        if self.log_to_wandb:
            self._log_to_wandb(stats, global_step)

    def _log_to_wandb(self, stats: Dict[str, Any], global_step: int) -> None:
        """Log stats to wandb if available."""
        # Lazy check for wandb availability
        if self._wandb_available is None:
            try:
                import wandb
                self._wandb_available = wandb.run is not None
            except ImportError:
                self._wandb_available = False

        if not self._wandb_available:
            return

        try:
            import wandb

            # Log aggregated metrics (peak and median values accumulated throughout the step)
            wandb.log(
                {
                    "vllm/num_engines": stats["num_engines"],
                    # Peak metrics
                    "vllm/peak_running_reqs": stats.get("total_peak_running_reqs", stats.get("total_running_reqs", 0)),
                    "vllm/peak_waiting_reqs": stats.get("total_peak_waiting_reqs", stats.get("total_waiting_reqs", 0)),
                    "vllm/peak_prompt_throughput": stats.get("avg_peak_prompt_throughput", stats.get("avg_prompt_throughput", 0.0)),
                    "vllm/peak_generation_throughput": stats.get("avg_peak_generation_throughput", stats.get("avg_generation_throughput", 0.0)),
                    "vllm/peak_gpu_cache_usage_perc": stats.get("avg_peak_gpu_cache_usage_perc", stats.get("avg_gpu_cache_usage_perc", 0.0)),
                    "vllm/peak_prefix_cache_hit_rate": stats.get("avg_peak_prefix_cache_hit_rate", stats.get("avg_prefix_cache_hit_rate", 0.0)),
                    # Median metrics
                    "vllm/median_running_reqs": stats.get("avg_median_running_reqs", 0.0),
                    "vllm/median_waiting_reqs": stats.get("avg_median_waiting_reqs", 0.0),
                    "vllm/median_prompt_throughput": stats.get("avg_median_prompt_throughput", 0.0),
                    "vllm/median_generation_throughput": stats.get("avg_median_generation_throughput", 0.0),
                    "vllm/median_gpu_cache_usage_perc": stats.get("avg_median_gpu_cache_usage_perc", 0.0),
                    "vllm/median_prefix_cache_hit_rate": stats.get("avg_median_prefix_cache_hit_rate", 0.0),
                    # Per-request latency (seconds)
                    "vllm/latency_prefill_mean": stats.get("avg_latency_prefill_mean", 0.0),
                    "vllm/latency_prefill_p90": stats.get("max_latency_prefill_p90", 0.0),
                    "vllm/latency_decode_mean": stats.get("avg_latency_decode_mean", 0.0),
                    "vllm/latency_decode_p90": stats.get("max_latency_decode_p90", 0.0),
                    "vllm/latency_e2e_mean": stats.get("avg_latency_e2e_mean", 0.0),
                    "vllm/latency_e2e_p90": stats.get("max_latency_e2e_p90", 0.0),
                    "vllm/latency_queued_mean": stats.get("avg_latency_queued_mean", 0.0),
                    "vllm/latency_queued_p90": stats.get("max_latency_queued_p90", 0.0),
                    "vllm/latency_ttft_mean": stats.get("avg_latency_ttft_mean", 0.0),
                    "vllm/latency_ttft_p90": stats.get("max_latency_ttft_p90", 0.0),
                    "vllm/total_finished_requests": stats.get("total_finished_requests", 0),
                    "vllm/total_preempted_reqs": stats.get("total_preempted_reqs", 0),
                    # Metadata
                    "vllm/total_samples": stats.get("total_samples", 0),
                    "vllm/total_active_samples": stats.get("total_active_samples", 0),
                },
                step=global_step,
            )

            # Also log per-engine metrics if there are multiple engines
            if stats["num_engines"] > 1:
                for i, engine_stats in enumerate(stats.get("engines", [])):
                    wandb.log(
                        {
                            # Peak metrics per engine
                            f"vllm/engine_{i}/peak_prompt_throughput": engine_stats.get(
                                "peak_prompt_throughput", engine_stats.get("avg_prompt_throughput", 0.0)
                            ),
                            f"vllm/engine_{i}/peak_generation_throughput": engine_stats.get(
                                "peak_generation_throughput", engine_stats.get("avg_generation_throughput", 0.0)
                            ),
                            f"vllm/engine_{i}/peak_running_reqs": engine_stats.get(
                                "peak_running_reqs", engine_stats.get("num_running_reqs", 0)
                            ),
                            f"vllm/engine_{i}/peak_waiting_reqs": engine_stats.get(
                                "peak_waiting_reqs", engine_stats.get("num_waiting_reqs", 0)
                            ),
                            f"vllm/engine_{i}/peak_gpu_cache_usage": engine_stats.get(
                                "peak_gpu_cache_usage_perc", engine_stats.get("gpu_cache_usage_perc", 0.0)
                            ),
                            # Median metrics per engine
                            f"vllm/engine_{i}/median_prompt_throughput": engine_stats.get("median_prompt_throughput", 0.0),
                            f"vllm/engine_{i}/median_generation_throughput": engine_stats.get("median_generation_throughput", 0.0),
                            f"vllm/engine_{i}/median_running_reqs": engine_stats.get("median_running_reqs", 0.0),
                            f"vllm/engine_{i}/median_waiting_reqs": engine_stats.get("median_waiting_reqs", 0.0),
                            # Per-engine latency stats
                            f"vllm/engine_{i}/latency_prefill_mean": engine_stats.get("latency_prefill_mean", 0.0),
                            f"vllm/engine_{i}/latency_prefill_p90": engine_stats.get("latency_prefill_p90", 0.0),
                            f"vllm/engine_{i}/latency_decode_mean": engine_stats.get("latency_decode_mean", 0.0),
                            f"vllm/engine_{i}/latency_decode_p90": engine_stats.get("latency_decode_p90", 0.0),
                            f"vllm/engine_{i}/latency_e2e_mean": engine_stats.get("latency_e2e_mean", 0.0),
                            f"vllm/engine_{i}/latency_e2e_p90": engine_stats.get("latency_e2e_p90", 0.0),
                            f"vllm/engine_{i}/latency_queued_mean": engine_stats.get("latency_queued_mean", 0.0),
                            f"vllm/engine_{i}/latency_ttft_mean": engine_stats.get("latency_ttft_mean", 0.0),
                            f"vllm/engine_{i}/finished_requests": engine_stats.get("latency_num_finished_requests", 0),
                            f"vllm/engine_{i}/preempted_reqs": engine_stats.get("total_preempted_reqs", 0),
                        },
                        step=global_step,
                    )
        except Exception as e:
            logger.warning(f"VLLMStatsCallback: Failed to log to wandb: {e}")


def create_default_callbacks(cfg: "DictConfig") -> List[TrainerCallback]:
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
    # Check for new-style explicit callback configuration
    callbacks_config = getattr(cfg.trainer, "callbacks", None)
    if callbacks_config is not None and len(callbacks_config) > 0:
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
    hf_save_interval = getattr(cfg.trainer, "hf_save_interval", -1)
    if hf_save_interval > 0:
        callbacks.append(HFModelSaveCallback(save_steps=hf_save_interval))

    # HF Hub upload callback (uploads saved HF models to HuggingFace Hub)
    hf_hub_repo_id = getattr(cfg.trainer, "hf_hub_repo_id", None)
    if hf_hub_repo_id and hf_save_interval > 0:
        hf_hub_private = getattr(cfg.trainer, "hf_hub_private", False)
        hf_hub_revision = getattr(cfg.trainer, "hf_hub_revision", "main")
        callbacks.append(
            HFHubUploadCallback(
                repo_id=hf_hub_repo_id,
                upload_steps=hf_save_interval,
                upload_on_train_end=True,
                private=hf_hub_private,
                revision=hf_hub_revision,
            )
        )

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

    # vLLM stats callback (enabled when using vLLM backend)
    # This collects engine stats directly, bypassing unreliable Ray log-to-driver
    generator_backend = getattr(cfg.generator, "backend", None)
    vllm_stats_interval = getattr(cfg.generator, "vllm_stats_interval", 1)
    if generator_backend == "vllm" and vllm_stats_interval > 0:
        callbacks.append(
            VLLMStatsCallback(
                log_every_steps=vllm_stats_interval,
                log_to_console=True,
                log_to_wandb=True,
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
        cfg: Optional["DictConfig"] = None,
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
        cfg: "DictConfig",
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
        raise ValueError(
            f"Unknown callback type '{callback_type}'. "
            f"Available types: {available}"
        )

    # Get the callback class and instantiate with remaining params
    callback_cls = CALLBACK_REGISTRY[callback_type]
    params = {k: v for k, v in callback_config.items() if k != "type"}

    try:
        return callback_cls(**params)
    except TypeError as e:
        raise ValueError(
            f"Invalid parameters for callback '{callback_type}': {e}"
        ) from e


def create_callbacks_from_config(cfg: "DictConfig") -> List[TrainerCallback]:
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

    def __init__(self, tracker: "DataConsumptionTracker"):
        from skyrl_train.utils.data_tracker import DataConsumptionTracker

        assert isinstance(tracker, DataConsumptionTracker)
        self._tracker = tracker

    def on_save(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        import dataclasses
        import os

        import torch

        from skyrl_train.utils.io import io

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
        tracker: "DataConsumptionTracker",
    ) -> bool:
        """Load data consumption state from a checkpoint directory.

        Tries the new data_consumption_state.pt first, then falls back to
        legacy fully_async_state.pt for backward compatibility.

        Returns True if state was loaded, False if no artifact found.
        """
        import os

        import torch

        from skyrl_train.utils.data_tracker import DataConsumptionState
        from skyrl_train.utils.io import io

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
                logger.info(
                    f"Loaded legacy fully_async_state.pt with {len(consumed)} consumed UIDs"
                )
                return True

        return False


class BufferCheckpointCallback(TrainerCallback):
    """Best-effort save/restore of the async generation buffer at checkpoint time.

    Saves all pending GeneratedOutputGroup items from the asyncio.Queue so that
    on resume the buffer can be restored without re-generating from scratch.
    """

    ARTIFACT_NAME = "generation_buffer_state.pt"
    error_behavior = "warn"

    def on_save(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        import os

        import torch

        from skyrl_train.utils.io import io

        trainer = kwargs.get("trainer")
        if trainer is None:
            logger.warning("BufferCheckpointCallback.on_save: no trainer in kwargs, skipping")
            return control

        buf = getattr(trainer, "_generation_output_group_buffer", None)
        if buf is None:
            return control

        try:
            # Drain-and-restore: non-destructive snapshot of the queue.
            # Safe because on_save runs synchronously within the event loop —
            # no generation worker can interleave between drain and restore.
            items = []
            while not buf.empty():
                try:
                    items.append(buf.get_nowait())
                except Exception:
                    break
            # Put them all back
            for item in items:
                buf.put_nowait(item)

            if not items:
                return control

            serialized = []
            for item in items:
                serialized.append({
                    "generator_output": dict(item.generator_output),
                    "uid": item.uid,
                    "global_step_when_scheduled": item.global_step_when_scheduled,
                })

            ckpt_path = os.path.join(
                trainer.cfg.trainer.ckpt_path,
                f"global_step_{state.global_step}",
            )
            artifact_path = os.path.join(ckpt_path, self.ARTIFACT_NAME)
            with io.open_file(artifact_path, "wb") as f:
                torch.save(serialized, f)
            logger.info(
                f"Saved {len(serialized)} generation buffer items to {artifact_path}"
            )
        except Exception as e:
            logger.warning(f"BufferCheckpointCallback.on_save failed (best-effort): {e}")

        return control

    @staticmethod
    def load_buffer_items(ckpt_path: str):
        """Load buffer items from a checkpoint directory.

        Returns a list of GeneratedOutputGroup, or empty list if no file found.
        """
        import os

        import torch

        from skyrl_train.fully_async_trainer import GeneratedOutputGroup
        from skyrl_train.generators.base import GeneratorOutput
        from skyrl_train.utils.io import io

        artifact_path = os.path.join(ckpt_path, BufferCheckpointCallback.ARTIFACT_NAME)
        if not io.exists(artifact_path):
            return []

        try:
            with io.open_file(artifact_path, "rb") as f:
                serialized = torch.load(f, map_location="cpu", weights_only=False)

            items = []
            for entry in serialized:
                gen_out: GeneratorOutput = entry["generator_output"]
                items.append(GeneratedOutputGroup(
                    generator_output=gen_out,
                    uid=entry["uid"],
                    global_step_when_scheduled=entry["global_step_when_scheduled"],
                ))
            return items
        except Exception as e:
            logger.warning(f"BufferCheckpointCallback.load_buffer_items failed: {e}")
            return []
