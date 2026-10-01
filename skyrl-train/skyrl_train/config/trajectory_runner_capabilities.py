"""Validate launch requirements for the shared task rollout engine."""

from enum import StrEnum

from omegaconf import DictConfig

from marinskyrl.distillation import compile_distillation_plan_from_config


class EntrypointOperation(StrEnum):
    TRAIN = "train"
    GENERATE = "generate"


def validate_trajectory_runner_capabilities(
    cfg: DictConfig,
    operation: EntrypointOperation = EntrypointOperation.TRAIN,
) -> None:
    """Reject launches that cannot supply the engine's exact-token transport."""
    if cfg.generator.backend != "vllm":
        raise ValueError("Task rollouts require generator.backend=vllm for exact structured-chat transport")
    if operation is EntrypointOperation.GENERATE and compile_distillation_plan_from_config(cfg) is not None:
        raise ValueError("teacher-scored distillation is training-only and cannot be configured for generation")
