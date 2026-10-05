"""Typed choices shared by the launcher and training runtime."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any


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
HARBOR_TOKEN_REWARD_DEFAULTS = MappingProxyType(
    {"enable_token_reward_channel": False, "enable_pbs_shaping": False, "enable_span_tagging": True}
)


def terminal_bench_config(cfg: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the configured terminal-bench settings for this entrypoint."""
    terminal_bench = cfg.get("terminal_bench_config")
    if terminal_bench is None and str(cfg.get("entrypoint", "")) == "terminal_bench":
        terminal_bench = cfg.get("terminal_bench")
    return terminal_bench


def pbs_token_credit_enabled(cfg: Mapping[str, Any]) -> bool:
    """Return whether the trainer and Harbor enable PBS token credit."""
    terminal_bench = terminal_bench_config(cfg)
    if terminal_bench is None or not cfg["trainer"]["algorithm"]["enable_token_reward_channel"]:
        return False
    harbor = terminal_bench.get("harbor") or {}
    for key, default in HARBOR_TOKEN_REWARD_DEFAULTS.items():
        value = harbor.get(key, terminal_bench.get(key, default))
        if not bool(default if value is None else value):
            return False
    return True


class RolloutGrading(StrEnum):
    """Launcher-side mirror of skyrl_gym's NemotronUltraGrading, which marinskyrl cannot import."""

    VERIFY = "verify"
    SKIP = "skip"


class AdvantageEstimator(StrEnum):
    GAE = "gae"
    GRPO = "grpo"
    RLOO = "rloo"
    RLOO_N = "rloo_n"  # RLOO with a configured minimum of baseline-eligible samples per group
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
