"""Typed choices shared by the launcher and training runtime."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class BatchBuilder(StrEnum):
    DRIVER = "driver"
    WORKER = "worker"


def parse_batch_builder(value: str) -> BatchBuilder:
    """Validate the public batch-builder choice at runtime configuration entry."""
    try:
        return BatchBuilder(value)
    except ValueError as error:
        raise ValueError("trainer.batch_builder must be driver or worker") from error


class R3Transport(StrEnum):
    BY_VALUE = "by_value"
    RESIDENT = "resident"
    DECENTRAL = "decentral"


class WeightSyncTransport(StrEnum):
    BROADCAST = "broadcast"
    EXPERT_BLOCK = "expert_block"


class GDNBackend(StrEnum):
    TORCH = "torch"
    FLASHQLA = "flashqla"


TRAJECTORY_SELECTOR_TYPE_PATH = "trainer.trajectory_selector.type"


class RolloutGrading(StrEnum):
    """Launcher-side mirror of skyrl_gym's NemotronUltraGrading, which marinskyrl cannot import."""

    VERIFY = "verify"
    SKIP = "skip"


class AdvantageEstimator(StrEnum):
    GAE = "gae"
    GRPO = "grpo"
    RLOO = "rloo"
    RLOO_N = "rloo_n"  # RLOO-Neutral: excludes masked samples from baseline
    REINFORCE_PP = "reinforce++"
    UNIFORM = "uniform"
    REWARD = "reward"


class PolicyLossType(StrEnum):
    REGULAR = "regular"
    DUAL_CLIP = "dual_clip"
    BEHAVIOR_CLIP = "behavior_clip"
    GSPO = "gspo"
    CISPO = "cispo"
    CLIP_COV = "clip_cov"
    KL_COV = "kl_cov"
    SAPO = "sapo"
    SFT = "sft"
    FTPO = "ftpo"
    IMPORTANCE_SAMPLING = "importance_sampling"


def reference_model_required(algorithm: Mapping[str, Any]) -> bool:
    """Return whether the objective needs a frozen reference actor."""
    return bool(
        algorithm.get("use_kl_loss")
        or algorithm.get("use_kl_in_reward")
        or algorithm.get("policy_loss_type") == PolicyLossType.FTPO
    )
