import json
import traceback
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional, Union

from loguru import logger

POSITIVE_RESPONSE_COLOR = "green"
NEGATIVE_RESPONSE_COLOR = "yellow"
BASE_PROMPT_COLOR = "cyan"
PROGRESS_LOG_PREFIX = "SKYRL_PROGRESS "


class ProgressEventName(StrEnum):
    SERVICE_STARTING = "service_starting"
    SERVICE_READY = "service_ready"
    ROLLOUT_BATCH_STARTED = "rollout_batch_started"
    ROLLOUT_BATCH_COMPLETED = "rollout_batch_completed"
    OPTIMIZER_STEP_COMPLETED = "optimizer_step_completed"
    WEIGHT_UPDATE_COMPLETED = "weight_update_completed"
    TRAINING_STEP_COMPLETED = "training_step_completed"


class ServiceName(StrEnum):
    INFERENCE_ENGINES = "inference_engines"
    POLICY_WORKERS = "policy_workers"
    TRAJECTORY_RUNNER = "trajectory_runner"
    WEIGHT_SYNC = "weight_sync"


class WeightUpdateReason(StrEnum):
    INITIAL = "initial"
    TRAINING_STEP = "training_step"
    CHECKPOINT_RESTORE = "checkpoint_restore"


@dataclass(frozen=True)
class ServiceStartingEvent:
    service: ServiceName
    implementation: str | None = None
    mode: str | None = None
    strategy: str | None = None
    event: ProgressEventName = field(default=ProgressEventName.SERVICE_STARTING, init=False)


@dataclass(frozen=True)
class ServiceReadyEvent:
    service: ServiceName
    implementation: str | None = None
    mode: str | None = None
    strategy: str | None = None
    count: int | None = None
    policy_workers: int | None = None
    inference_engines: int | None = None
    event: ProgressEventName = field(default=ProgressEventName.SERVICE_READY, init=False)


@dataclass(frozen=True)
class RolloutBatchStartedEvent:
    step: int
    mode: str
    prompts: int | None = None
    required_groups: int | None = None
    event: ProgressEventName = field(default=ProgressEventName.ROLLOUT_BATCH_STARTED, init=False)


@dataclass(frozen=True)
class RolloutBatchCompletedEvent:
    step: int
    mode: str
    duration_seconds: float
    trajectories: int
    response_tokens: int
    prompts: int | None = None
    groups: int | None = None
    staleness_mean: float | None = None
    staleness_max: int | None = None
    event: ProgressEventName = field(default=ProgressEventName.ROLLOUT_BATCH_COMPLETED, init=False)


@dataclass(frozen=True)
class OptimizerStepCompletedEvent:
    step: int
    epoch: int
    sequences: int
    duration_seconds: float
    event: ProgressEventName = field(default=ProgressEventName.OPTIMIZER_STEP_COMPLETED, init=False)


@dataclass(frozen=True)
class WeightUpdateCompletedEvent:
    step: int | None
    reason: WeightUpdateReason
    duration_seconds: float
    event: ProgressEventName = field(default=ProgressEventName.WEIGHT_UPDATE_COMPLETED, init=False)


@dataclass(frozen=True)
class TrainingStepCompletedEvent:
    step: int
    epoch: int
    total_steps: int
    duration_seconds: float
    event: ProgressEventName = field(default=ProgressEventName.TRAINING_STEP_COMPLETED, init=False)


ProgressEvent = (
    ServiceStartingEvent
    | ServiceReadyEvent
    | RolloutBatchStartedEvent
    | RolloutBatchCompletedEvent
    | OptimizerStepCompletedEvent
    | WeightUpdateCompletedEvent
    | TrainingStepCompletedEvent
)


def format_progress_event(progress: ProgressEvent) -> str:
    """Format one machine-readable application progress event for text log capture."""
    payload = {key: value for key, value in asdict(progress).items() if value is not None}
    return PROGRESS_LOG_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def log_progress(progress: ProgressEvent) -> None:
    """Log a machine-readable application progress event at INFO level."""
    logger.opt(depth=1).info(format_progress_event(progress))


def format_exception_text(error: BaseException) -> str:
    """Format an exception without retaining it in a queued logging record."""
    return "".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip()


def log_exception_as_text(context: str, error: BaseException) -> None:
    """Emit a traceback as text so Loguru never needs to pickle the exception."""
    logger.error("{}:\n{}", context, format_exception_text(error))


def _color_block_format_and_kwargs(
    text: str,
    color: str,
    field_prefix: str,
) -> tuple[str, dict]:
    """Build a format string and kwargs for a multi-line colored block.

    The format string will look like:
        "<color>{p0}</color>\n<color>{p1}</color>\n..."

    where "p0", "p1", ... are placeholder names starting with `field_prefix`.
    """
    # Ensure at least one line
    lines = text.splitlines() or [""]

    fmt_lines = []
    kwargs: dict[str, str] = {}

    for i, line in enumerate(lines):
        key = f"{field_prefix}{i}"
        # NOTE: double braces {{ }} so that {key} survives into str.format
        fmt_lines.append(f"<{color}>{{{key}}}</{color}>")
        kwargs[key] = line

    fmt = "\n".join(fmt_lines)
    return fmt, kwargs


def log_example(
    logger: Any,
    prompt: List[Dict[str, Any]],
    response: str,
    reward: Optional[Union[float, List[float]]] = None,
) -> None:
    """
    Log a single example prompt and response with formatting and colors.

    Args:
        logger: The logger instance to use (expected to be loguru logger or compatible).
        prompt: The input prompt in OpenAI message format.
        response: The output response string.
        reward: The reward value(s) associated with the response.
    """
    reward_val = 0.0
    reward_str = "N/A"
    try:
        prompt_str = str(prompt)
        response_str = str(response)
        # --- Reward handling ---
        if reward is not None:
            if isinstance(reward, list):
                reward_val = float(sum(reward))
            else:
                reward_val = float(reward)
            reward_str = f"{reward_val:.4f}"

        # --- Color selection ---
        if reward is not None and reward_val > 0:
            response_color = POSITIVE_RESPONSE_COLOR
        else:
            response_color = NEGATIVE_RESPONSE_COLOR

        # --- Build per-line colored blocks in the *format string* ---
        prompt_fmt, prompt_kwargs = _color_block_format_and_kwargs(prompt_str, BASE_PROMPT_COLOR, "p")
        response_fmt, response_kwargs = _color_block_format_and_kwargs(response_str, response_color, "r")

        # Single format string with only our own markup and placeholders
        log_format = f"Example:\n  Input: {prompt_fmt}\n  Output (Total Reward: {{reward}}):\n{response_fmt}"

        # Merge all args for str.format
        format_kwargs = {**prompt_kwargs, **response_kwargs, "reward": reward_str}

        # Let Loguru parse tags in log_format and then substitute arguments.
        logger.opt(colors=True).info(log_format, **format_kwargs)
    except Exception as e:
        logger.warning("Error pretty printing example; logging plain text instead: {}", e)
        logger.info(
            "Example:\n  Input: {}\n  Output (Total Reward: {}):\n{}",
            prompt,
            reward_str,
            response,
        )
