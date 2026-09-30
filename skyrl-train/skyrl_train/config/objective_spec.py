"""Torch-free contracts for policy objectives and their configuration."""

import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from omegaconf import DictConfig, OmegaConf

from marinskyrl.distillation import (
    DistillationObjectiveKind,
    DistillationRewardMode,
    compile_distillation_plan_from_config,
)

from marinskyrl.runtime_options import AdvantageEstimator, PolicyLossType


class KLEstimator(StrEnum):
    K1 = "k1"
    ABS = "abs"
    K2 = "k2"
    K3 = "k3"
    K3_UNBIASED_GRADIENT = "k3_unbiased_gradient"


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
        PolicyLossType.REGULAR: LossSpec(RatioAnchor.OLD),
        PolicyLossType.DUAL_CLIP: LossSpec(RatioAnchor.OLD),
        PolicyLossType.GSPO: LossSpec(RatioAnchor.OLD, sequence_level=True),
        PolicyLossType.CISPO: LossSpec(RatioAnchor.OLD),
        PolicyLossType.SAPO: LossSpec(RatioAnchor.OLD),
        PolicyLossType.CLIP_COV: LossSpec(RatioAnchor.OLD, row_local=False),
        PolicyLossType.KL_COV: LossSpec(RatioAnchor.OLD, row_local=False, advantage_linear=False),
        PolicyLossType.IMPORTANCE_SAMPLING: LossSpec(RatioAnchor.OLD),
        PolicyLossType.BEHAVIOR_CLIP: LossSpec(RatioAnchor.ROLLOUT),
        PolicyLossType.SFT: LossSpec(RatioAnchor.NONE, advantage_linear=False),
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


def topk_loss_params(algorithm: DictConfig) -> TopKLossParams:
    """Compute the teacher-loss parameters from an active top-K objective config."""
    return TopKLossParams(
        DistillationObjectiveKind(algorithm.distillation.objective),
        float(algorithm.eps_clip_low),
        float(algorithm.eps_clip_high),
        float(algorithm.clip_ratio_c),
    )


def rollout_logprobs_required(algorithm: DictConfig, *, loss_spec: LossSpec | None = None) -> bool:
    """Compute whether active policy rows require behavior-policy log probabilities."""
    spec = loss_spec or BUILTIN_LOSS_SPECS.get(algorithm.policy_loss_type)
    if spec is None:
        raise ValueError(f"policy loss {algorithm.policy_loss_type!r} requires a runtime LossSpec")
    distillation = algorithm.get("distillation")
    if (
        distillation is not None
        and distillation.objective != DistillationObjectiveKind.SAMPLED_REVERSE_KL
        and distillation.reward_mode == DistillationRewardMode.REPLACE
    ):
        return False
    return spec.anchor is RatioAnchor.ROLLOUT or bool(algorithm.use_tis)


def validate_objective(cfg: DictConfig, *, loss_spec: LossSpec | None = None) -> None:
    """Reject objective settings that cannot affect the selected training rows correctly."""
    limit = cfg.trainer.policy.max_consecutive_nonfinite_steps
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("trainer.policy.max_consecutive_nonfinite_steps must be null or an integer >= 1")
    algorithm = cfg.trainer.algorithm
    for key in ("use_abs_kl", "use_kl_estimator_k3"):
        if key in algorithm:
            raise ValueError(f"trainer.algorithm.{key} is unsupported; configure trainer.algorithm.kl_estimator_type")
    try:
        KLEstimator(algorithm.kl_estimator_type)
    except ValueError as error:
        raise ValueError(
            f"invalid kl_estimator_type: {algorithm.kl_estimator_type}; choose one of {list(KLEstimator)}"
        ) from error
    try:
        reduction = LossReduction(algorithm.loss_reduction)
    except ValueError as error:
        raise ValueError(
            f"invalid loss_reduction: {algorithm.loss_reduction}; choose one of {list(LossReduction)}"
        ) from error
    spec = loss_spec or BUILTIN_LOSS_SPECS.get(algorithm.policy_loss_type)
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
    if not topk and algorithm.policy_loss_type == PolicyLossType.SFT:
        raise ValueError(
            "sampled_reverse_kl requires a policy loss that consumes advantages; sft ignores teacher credit"
        )
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
        if algorithm.advantage_estimator != AdvantageEstimator.UNIFORM:
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
