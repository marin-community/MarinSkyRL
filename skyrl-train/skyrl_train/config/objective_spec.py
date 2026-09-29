"""Torch-free contracts for policy objectives and their configuration."""

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from omegaconf import DictConfig, OmegaConf

from marinskyrl.distillation import (
    DistillationObjectiveKind,
    DistillationRewardMode,
    compile_distillation_plan_from_config,
)


class RatioAnchor(StrEnum):
    OLD = "old"
    ROLLOUT = "rollout"
    NONE = "none"


@dataclass(frozen=True)
class LossSpec:
    anchor: RatioAnchor
    sequence_level: bool = False
    row_local: bool = True
    advantage_linear: bool = True


BUILTIN_LOSS_SPECS: Mapping[str, LossSpec] = MappingProxyType(
    {
        "regular": LossSpec(RatioAnchor.OLD),
        "dual_clip": LossSpec(RatioAnchor.OLD),
        "gspo": LossSpec(RatioAnchor.OLD, sequence_level=True),
        "cispo": LossSpec(RatioAnchor.OLD),
        "sapo": LossSpec(RatioAnchor.OLD),
        "clip_cov": LossSpec(RatioAnchor.OLD, row_local=False),
        "kl_cov": LossSpec(RatioAnchor.OLD, row_local=False, advantage_linear=False),
        "importance_sampling": LossSpec(RatioAnchor.OLD),
        "behavior_clip": LossSpec(RatioAnchor.ROLLOUT),
        "sft": LossSpec(RatioAnchor.NONE, advantage_linear=False),
    }
)


class LossReduction(StrEnum):
    TOKEN_MEAN = "token_mean"
    SEQUENCE_MEAN = "sequence_mean"
    SEQ_MEAN_TOKEN_SUM_NORM = "seq_mean_token_sum_norm"
    SEQ_MEAN_TOKEN_SUM_NORM_GLOBAL = "seq_mean_token_sum_norm_global"


@dataclass(frozen=True)
class TopKLossParams:
    objective: DistillationObjectiveKind
    eps_clip_low: float
    eps_clip_high: float
    clip_ratio_c: float

    @classmethod
    def from_config(cls, config: DictConfig) -> "TopKLossParams":
        return cls(
            DistillationObjectiveKind(config.objective),
            float(config.eps_clip_low),
            float(config.eps_clip_high),
            float(config.clip_ratio_c),
        )


def _loss_spec(algorithm: DictConfig) -> LossSpec | None:
    name = algorithm.policy_loss_type
    if name in BUILTIN_LOSS_SPECS:
        return BUILTIN_LOSS_SPECS[name]
    resolved = algorithm.get("resolved_loss_spec")
    if resolved is None:
        return None
    return LossSpec(
        anchor=RatioAnchor(resolved.anchor),
        sequence_level=resolved.sequence_level,
        row_local=resolved.row_local,
        advantage_linear=resolved.advantage_linear,
    )


def resolve_objective_config(cfg: DictConfig) -> None:
    """Materialize objective contracts as primitive config for driver and worker transport."""
    algorithm = cfg.trainer.algorithm
    spec = _loss_spec(algorithm)
    if spec is None:
        raise ValueError(f"policy loss {algorithm.policy_loss_type!r} requires a runtime LossSpec")
    algorithm.resolved_loss_spec = {**asdict(spec), "anchor": spec.anchor.value}
    plan = compile_distillation_plan_from_config(cfg)
    topk = plan is not None and plan.objective is not DistillationObjectiveKind.SAMPLED_REVERSE_KL
    algorithm.resolved_topk_loss_params = (
        {
            "objective": plan.objective.value,
            "eps_clip_low": float(algorithm.eps_clip_low),
            "eps_clip_high": float(algorithm.eps_clip_high),
            "clip_ratio_c": float(algorithm.clip_ratio_c),
        }
        if topk
        else None
    )
    policy_trains = not (topk and plan.reward_mode is DistillationRewardMode.REPLACE)
    algorithm.resolved_rollout_logprobs_required = bool(
        (policy_trains and (spec.anchor is RatioAnchor.ROLLOUT or algorithm.use_tis))
        or (plan is not None and plan.objective is DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE)
    )


def validate_objective(cfg: DictConfig) -> None:
    """Reject objective settings that cannot affect the selected training rows correctly."""
    limit = cfg.trainer.policy.max_consecutive_nonfinite_steps
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("trainer.policy.max_consecutive_nonfinite_steps must be null or an integer >= 1")
    algorithm = cfg.trainer.algorithm
    try:
        reduction = LossReduction(algorithm.loss_reduction)
    except ValueError as error:
        raise ValueError(
            f"invalid loss_reduction: {algorithm.loss_reduction}; choose one of {list(LossReduction)}"
        ) from error
    spec = _loss_spec(algorithm)
    plan = compile_distillation_plan_from_config(cfg)
    topk = plan is not None and plan.objective is not DistillationObjectiveKind.SAMPLED_REVERSE_KL
    loop_credit = float(
        OmegaConf.select(cfg, "generator.trajectory_reward_shaping.loop.advantage_penalty_per_token", default=0)
    )
    if spec is not None and spec.sequence_level:
        if reduction is not LossReduction.SEQUENCE_MEAN:
            raise ValueError(f"{algorithm.policy_loss_type} requires trainer.algorithm.loss_reduction=sequence_mean")
        if loop_credit > 0 or (plan is not None and not topk):
            raise ValueError(f"{algorithm.policy_loss_type} requires sequence-level advantages; use a token-level loss")
    if algorithm.think_token_weight != 1 and not algorithm.enable_token_reward_channel:
        raise ValueError("think_token_weight != 1 requires trainer.algorithm.enable_token_reward_channel=true")
    if plan is None:
        return
    if topk:
        policy = cfg.trainer.policy
        if (
            cfg.trainer.use_sample_packing
            or policy.sequence_parallel_size != 1
            or policy.megatron_config.context_parallel_size != 1
            or policy.megatron_config.tensor_model_parallel_size != 1
        ):
            raise ValueError(
                "top-K teacher objectives require use_sample_packing=false, sequence_parallel_size=1, "
                "context_parallel_size=1, and tensor_model_parallel_size=1"
            )
        if reduction is LossReduction.SEQ_MEAN_TOKEN_SUM_NORM_GLOBAL:
            raise ValueError("top-K teacher rows require token_mean, sequence_mean, or seq_mean_token_sum_norm")
        if plan.advantage_clip is not None:
            raise ValueError("advantage_clip requires sampled_reverse_kl chosen-token evidence")
    if plan.objective is DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE:
        if reduction is not LossReduction.TOKEN_MEAN:
            raise ValueError("student_topk_policy_surrogate requires token_mean loss reduction")
        low, high, dual = float(algorithm.eps_clip_low), float(algorithm.eps_clip_high), float(algorithm.clip_ratio_c)
        if not all(math.isfinite(value) for value in (low, high, dual)) or not (
            0 <= low < 1 and high >= 0 and dual > 1
        ):
            raise ValueError(
                "student_topk_policy_surrogate requires 0 <= eps_clip_low < 1, eps_clip_high >= 0, clip_ratio_c > 1"
            )
    if plan.reward_mode is DistillationRewardMode.REPLACE:
        if spec is not None and not spec.advantage_linear:
            raise ValueError("distillation reward_mode=replace requires an advantage-linear policy loss")
        if algorithm.advantage_estimator != "uniform":
            raise ValueError("distillation reward_mode=replace requires advantage_estimator=uniform")
        if (
            algorithm.use_kl_in_reward
            or algorithm.advantage_batch_normalize
            or loop_credit > 0
            or algorithm.dynamic_sampling.type is not None
        ):
            raise ValueError(
                "distillation reward_mode=replace requires use_kl_in_reward=false, advantage_batch_normalize=false, "
                "loop.advantage_penalty_per_token=0 and dynamic_sampling.type=null"
            )
