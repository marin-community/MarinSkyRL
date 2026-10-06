"""Compile the controlled PivotRL/SFT comparison onto the standard trainer."""

from copy import deepcopy
from typing import Any


def apply_pivot_mode(raw: dict[str, Any]) -> dict[str, Any]:
    """Compile sampled GRPO or teacher-forced SFT on the frozen data."""
    if "pivot" not in raw:
        return raw
    raw = deepcopy(raw)
    mode = raw["pivot"]["mode"]
    raw["data"]["prompt_length_policy"] = "keep"
    if mode not in {"pivotrl", "sft", "sft_random"}:
        raise ValueError("pivot.mode must be pivotrl, sft, or sft_random")
    trainer, generator = raw["trainer"], raw["generator"]
    algorithm = trainer["algorithm"]
    if raw["context_budget"]["max_turns"] != 1:
        raise ValueError("Pivot experiments require one sampled action per rollout")
    if generator.get("trajectory_reward_shaping", {}).get("enabled", False):
        raise ValueError("Pivot experiments require binary rewards without shaping")
    if algorithm.get("dynamic_sampling", {}).get("type") is not None:
        raise ValueError("Pivot selection must be frozen before training")
    generator["reference_actions"] = mode in {"sft", "sft_random"}
    algorithm["use_kl_in_reward"] = False
    if mode in {"sft", "sft_random"}:
        algorithm.update(
            policy_loss_type="sft",
            advantage_estimator="uniform",
            use_kl_loss=False,
            kl_loss_coef=0.0,
            use_entropy_loss=False,
            off_policy_correction="none",
            group_advantage_min_size=None,
        )
        generator["n_samples_per_prompt"] = 1
    elif mode == "pivotrl":
        # Keep reference placement active even for beta=0, preserving the experiment topology.
        algorithm.update(
            policy_loss_type="regular",
            advantage_estimator="grpo",
            use_kl_loss=True,
            kl_estimator_type="forward",
            grpo_norm_by_std=True,
            group_advantage_min_size=None,
        )
    return raw
