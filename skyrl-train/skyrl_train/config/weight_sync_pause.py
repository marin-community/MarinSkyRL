"""Policy for pausing vLLM generation during a weight sync."""

from dataclasses import dataclass
from enum import StrEnum

from omegaconf import DictConfig


class WeightSyncPauseMode(StrEnum):
    """How vLLM handles in-flight requests during a weight sync."""

    ABORT = "abort"
    WAIT = "wait"
    KEEP = "keep"


@dataclass(frozen=True)
class WeightSyncPausePolicy:
    """Pause mode and cache treatment for a vLLM weight sync."""

    mode: WeightSyncPauseMode
    clear_cache: bool


DEFAULT_WEIGHT_SYNC_PAUSE_POLICY = WeightSyncPausePolicy(WeightSyncPauseMode.KEEP, True)


def validate_weight_sync_pause_backend(policy: WeightSyncPausePolicy, *, backend: str) -> None:
    """Restrict configured pause behavior to vLLM-compatible engines."""
    if policy != DEFAULT_WEIGHT_SYNC_PAUSE_POLICY and backend != "vllm":
        raise ValueError("non-default generator.weight_sync_pause requires vLLM inference engines")


def resolve_weight_sync_pause_policy(generator: DictConfig) -> WeightSyncPausePolicy:
    """Validate the configured pause policy against the inference engine topology."""
    raw_policy = generator.weight_sync_pause
    try:
        mode = WeightSyncPauseMode(raw_policy.mode)
    except ValueError as error:
        raise ValueError(
            f"generator.weight_sync_pause.mode must be abort, wait, or keep; got {raw_policy.mode!r}"
        ) from error
    if not isinstance(raw_policy.clear_cache, bool):
        raise ValueError("generator.weight_sync_pause.clear_cache must be a boolean")

    policy = WeightSyncPausePolicy(mode=mode, clear_cache=raw_policy.clear_cache)
    validate_weight_sync_pause_backend(policy, backend=generator.backend)
    if not policy.clear_cache and policy.mode is not WeightSyncPauseMode.KEEP:
        raise ValueError("generator.weight_sync_pause.clear_cache=false requires mode=keep")
    if policy.mode is WeightSyncPauseMode.WAIT and generator.vllm_v1_disable_multiproc:
        raise ValueError("generator.weight_sync_pause.mode=wait requires vllm_v1_disable_multiproc=false")
    return policy
