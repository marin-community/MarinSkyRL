"""Torch-free contracts for policy objectives and their configuration."""

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
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


class CorrectionAction(StrEnum):
    TRUNCATE = "truncate"
    MASK = "mask"


class SequenceAggregate(StrEnum):
    GEOMETRIC = "geometric"
    PRODUCT = "product"
    EXTREME_TOKEN = "extreme_token"


@dataclass(frozen=True)
class TokenRule:
    action: CorrectionAction
    low: float | None
    high: float | None


@dataclass(frozen=True)
class SequenceRule:
    aggregate: SequenceAggregate
    action: CorrectionAction
    low: float | None
    high: float | None


@dataclass(frozen=True)
class OffPolicyCorrection:
    name: str
    rules: tuple[TokenRule | SequenceRule, ...]

    @classmethod
    def from_config(cls, config: DictConfig) -> "OffPolicyCorrection":
        rules = []
        for rule in config.rules:
            kind = rule["kind"]
            allowed = {"kind", "action", "low", "high"}
            if kind == "sequence":
                allowed.add("aggregate")
            unknown = set(rule) - allowed
            if unknown:
                raise ValueError(f"unknown off_policy_correction rule fields: {sorted(unknown)}")
            action = CorrectionAction(rule["action"])
            low, high = rule.get("low"), rule.get("high")
            for bound in (low, high):
                if bound is not None and (
                    isinstance(bound, bool)
                    or not isinstance(bound, (int, float))
                    or not math.isfinite(bound)
                    or bound <= 0
                ):
                    raise ValueError("off_policy_correction bounds must be positive finite numbers")
            if low is not None and high is not None and low > high:
                raise ValueError("off_policy_correction requires low <= high")
            if action is CorrectionAction.TRUNCATE and (low is not None or high is None):
                raise ValueError("off_policy_correction truncate requires high and low=null")
            if action is CorrectionAction.MASK and low is None and high is None:
                raise ValueError("off_policy_correction mask requires low or high")
            if kind == "token":
                rules.append(TokenRule(action, low, high))
            elif kind == "sequence":
                aggregate = SequenceAggregate(rule["aggregate"])
                if aggregate is SequenceAggregate.EXTREME_TOKEN and action is not CorrectionAction.MASK:
                    raise ValueError("off_policy_correction extreme_token requires action=mask")
                rules.append(SequenceRule(aggregate, action, low, high))
            else:
                raise ValueError("off_policy_correction rule kind must be token or sequence")
        if sum(rule.action is CorrectionAction.TRUNCATE for rule in rules) > 1:
            raise ValueError("off_policy_correction permits at most one truncate rule")
        return cls(str(config.name), tuple(rules))

    def to_config(self) -> dict:
        rules = []
        for rule in self.rules:
            value = {"kind": "token", "action": rule.action.value, "low": rule.low, "high": rule.high}
            if isinstance(rule, SequenceRule):
                value.update(kind="sequence", aggregate=rule.aggregate.value)
            rules.append(value)
        return {"name": self.name, "rules": rules}


def load_correction(name: str) -> OffPolicyCorrection:
    """Read a named correction's rules without importing the training runtime."""
    if name == "none":
        return OffPolicyCorrection(name, ())
    if name not in {"tis", "icepop", "seq_mask_tis", "outlier_mask"}:
        raise ValueError(f"unknown off_policy_correction {name!r}; use a preset, none, null, or custom with rules")
    config = OmegaConf.load(Path(__file__).parent / "off_policy_correction" / f"{name}.yaml")
    return OffPolicyCorrection.from_config(OmegaConf.create({"name": name, "rules": config.rules}))


def _configured_correction(algorithm: DictConfig) -> OffPolicyCorrection:
    name = algorithm.off_policy_correction
    rules = algorithm.off_policy_correction_rules
    if name == "custom":
        if not rules:
            raise ValueError("off_policy_correction=custom requires off_policy_correction_rules")
        return OffPolicyCorrection.from_config(OmegaConf.create({"name": name, "rules": rules}))
    if rules:
        raise ValueError("off_policy_correction_rules requires off_policy_correction=custom")
    return load_correction("none" if name is None else str(name))


@dataclass(frozen=True)
class TopKLossParams:
    objective: DistillationObjectiveKind
    eps_clip_low: float
    eps_clip_high: float
    clip_ratio_c: float
    jsd_beta: float | None = None
    entry_clip: float | None = None

    @classmethod
    def from_config(cls, config: DictConfig) -> "TopKLossParams":
        return cls(
            DistillationObjectiveKind(config.objective),
            float(config.eps_clip_low),
            float(config.eps_clip_high),
            float(config.clip_ratio_c),
            config.jsd_beta,
            config.entry_clip,
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
    correction = _configured_correction(algorithm)
    plan = compile_distillation_plan_from_config(cfg)
    topk = plan is not None and plan.objective is not DistillationObjectiveKind.SAMPLED_REVERSE_KL
    algorithm.resolved_topk_loss_params = (
        {
            "objective": plan.objective.value,
            "jsd_beta": plan.jsd_beta,
            "entry_clip": plan.entry_clip,
            "eps_clip_low": float(algorithm.eps_clip_low),
            "eps_clip_high": float(algorithm.eps_clip_high),
            "clip_ratio_c": float(algorithm.clip_ratio_c),
        }
        if topk
        else None
    )
    policy_trains = not (topk and plan.reward_mode is DistillationRewardMode.REPLACE)
    algorithm.resolved_off_policy_correction = correction.to_config()
    algorithm.resolved_rollout_logprobs_required = bool(
        policy_trains and (spec.anchor is RatioAnchor.ROLLOUT or correction.rules)
    )


def validate_objective(cfg: DictConfig) -> None:
    """Reject objective settings that cannot affect the selected training rows correctly."""
    limit = cfg.trainer.policy.max_consecutive_nonfinite_steps
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("trainer.policy.max_consecutive_nonfinite_steps must be null or an integer >= 1")
    algorithm = cfg.trainer.algorithm
    for key in ("use_tis", "tis_imp_ratio_cap"):
        if key in algorithm:
            raise ValueError(
                f"trainer.algorithm.{key} is unsupported; configure trainer.algorithm.off_policy_correction"
            )
    try:
        reduction = LossReduction(algorithm.loss_reduction)
    except ValueError as error:
        raise ValueError(
            f"invalid loss_reduction: {algorithm.loss_reduction}; choose one of {list(LossReduction)}"
        ) from error
    spec = _loss_spec(algorithm)
    correction = _configured_correction(algorithm)
    plan = compile_distillation_plan_from_config(cfg)
    topk = plan is not None and plan.objective is not DistillationObjectiveKind.SAMPLED_REVERSE_KL
    policy_trains = not (topk and plan.reward_mode is DistillationRewardMode.REPLACE)
    if correction.rules:
        if not policy_trains:
            raise ValueError("off_policy_correction requires an active policy row; use none with top-K REPLACE")
        if spec is not None and spec.anchor is not RatioAnchor.OLD:
            raise ValueError("off_policy_correction requires an OLD-anchored policy loss; use none for this loss")
    if (
        cfg.trainer.rollout_buffer.max_staleness_steps > 0
        and policy_trains
        and spec is not None
        and spec.anchor is RatioAnchor.OLD
        and algorithm.off_policy_correction is None
    ):
        raise ValueError(
            "off-policy OLD-anchored training requires off_policy_correction; use 'none' to run without one"
        )
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
    if not topk and algorithm.policy_loss_type == "sft":
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
        widths = {teacher.top_k for teacher in plan.teachers}
        if widths != {cfg.generator.sampling_params.logprobs}:
            raise ValueError("student_topk_policy_surrogate requires sampling_params.logprobs matching teacher top_k")
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
