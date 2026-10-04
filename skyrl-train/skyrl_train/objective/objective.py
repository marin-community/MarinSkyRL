from dataclasses import dataclass

import torch

from marinskyrl.distillation import DistillationRewardMode
from omegaconf import DictConfig, OmegaConf

from skyrl_train.config.objective_spec import LossReduction, TopKLossParams, score_centering_tis_cap
from skyrl_train.ftpo import FTPOInputs
from skyrl_train.objective.losses import PolicyLoss, PolicyLossInputs, TokenLoss, complete_clip_metrics
from skyrl_train.objective.reduction import StepCounts, policy_data_weights, reduce_to_step
from skyrl_train.objective.score_centering import masked_topk_tail_mass, ppo_tis_score_centering_correction
from skyrl_train.objective.teacher import TopKEvidence, mask_teacher_evidence, topk_teacher_loss
from skyrl_train.utils.policy_math import differentiable_approx_kl
from skyrl_train.tensor_math import masked_mean


@dataclass(frozen=True)
class TopKTeacherBatch:
    evidence: TopKEvidence
    student_log_probs_on_support: torch.Tensor
    params: TopKLossParams
    vocabulary_size: int


@dataclass(frozen=True)
class ScoreCenteringBatch:
    current_log_probs: torch.Tensor
    old_log_probs: torch.Tensor
    behavior_log_probs: torch.Tensor


@dataclass(frozen=True)
class ObjectiveMicroBatch:
    policy: PolicyLossInputs
    policy_data_weights: torch.Tensor
    ref_log_probs: torch.Tensor | None
    token_entropy: torch.Tensor
    teacher: TopKTeacherBatch | None
    correction_weights: torch.Tensor | None = None
    score_centering: ScoreCenteringBatch | None = None


@dataclass(frozen=True)
class ObjectiveRows:
    policy: torch.Tensor
    kl: torch.Tensor
    entropy: torch.Tensor
    teacher: torch.Tensor | None


@dataclass(frozen=True)
class PolicyObjective:
    optimization_loss: torch.Tensor
    rows: ObjectiveRows
    metrics: dict[str, float]


def build_objective_micro_batch(
    *,
    action_log_probs: torch.Tensor,
    old_action_log_probs: torch.Tensor,
    base_action_log_probs: torch.Tensor | None,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    rollout_logprobs: torch.Tensor | None,
    response_span_tags: torch.Tensor | None,
    token_entropy: torch.Tensor,
    think_token_weight: float,
    teacher: TopKTeacherBatch | None,
    ftpo: FTPOInputs | None = None,
    correction_weights: torch.Tensor | None = None,
    score_centering: ScoreCenteringBatch | None = None,
) -> ObjectiveMicroBatch:
    """Prepare finite values at masked positions before objective formulas run."""
    valid = loss_mask > 0

    def sanitize(value: torch.Tensor) -> torch.Tensor:
        assert value.shape == loss_mask.shape
        return torch.where(valid, value, 0)

    if teacher is not None:
        teacher = TopKTeacherBatch(
            mask_teacher_evidence(teacher.evidence, loss_mask),
            teacher.student_log_probs_on_support,
            teacher.params,
            teacher.vocabulary_size,
        )
    return ObjectiveMicroBatch(
        policy=PolicyLossInputs(
            log_probs=sanitize(action_log_probs),
            old_log_probs=sanitize(old_action_log_probs),
            rollout_log_probs=None if rollout_logprobs is None else sanitize(rollout_logprobs),
            advantages=sanitize(advantages),
            loss_mask=loss_mask,
            ftpo=ftpo,
        ),
        policy_data_weights=policy_data_weights(loss_mask, response_span_tags, think_token_weight),
        ref_log_probs=None if base_action_log_probs is None else sanitize(base_action_log_probs),
        token_entropy=sanitize(token_entropy),
        teacher=teacher,
        correction_weights=None if correction_weights is None else sanitize(correction_weights).detach(),
        score_centering=score_centering,
    )


def compute_policy_objective(
    batch: ObjectiveMicroBatch,
    *,
    loss: PolicyLoss,
    counts: StepCounts,
    config: DictConfig,
    loss_scale: float,
    report_scale: float,
) -> PolicyObjective:
    """Compose globally normalized rows with separate backward and reporting scales."""
    teacher_only = (
        batch.teacher is not None
        and OmegaConf.select(config, "distillation.reward_mode") == DistillationRewardMode.REPLACE
    )
    policy = TokenLoss(batch.policy.log_probs * 0, {}) if teacher_only else loss(batch.policy, config)
    assert policy.values.shape == batch.policy.log_probs.shape
    common = {"max_seq_len": counts.max_seq_len, "nonzero_advantage_rows": counts.nonzero_advantage_rows}
    mode = LossReduction(config.loss_reduction)
    policy_row = reduce_to_step(
        policy.values,
        batch.policy_data_weights,
        counts.policy,
        mode,
        numerator_weights=batch.correction_weights,
        **common,
    )
    centering_metrics = {}
    if config.get("score_centering_topk", 0) and config.get("score_centering_enabled", True):
        evidence = batch.score_centering
        if evidence is None:
            raise ValueError("score centering requires aligned current, old, and behavior top-K log probabilities")
        centering = ppo_tis_score_centering_correction(
            evidence.current_log_probs,
            evidence.old_log_probs,
            evidence.behavior_log_probs,
            batch.policy.advantages,
            batch.policy.loss_mask,
            tis_cap=score_centering_tis_cap(config),
            eps_clip_low=config.eps_clip_low,
            eps_clip_high=config.eps_clip_high,
        )
        # Centering already integrates TIS over the behavior distribution. Applying the
        # sampled-action correction weight again would change its gradient.
        policy_row = policy_row + reduce_to_step(centering, batch.policy_data_weights, counts.policy, mode, **common)
        centering_metrics["score_centering/correction_abs_mean"] = masked_mean(
            centering.detach().abs(), batch.policy_data_weights
        ).item()
        for name, log_probs in (
            ("current", evidence.current_log_probs),
            ("old", evidence.old_log_probs),
            ("behavior", evidence.behavior_log_probs),
        ):
            tail = masked_topk_tail_mass(log_probs, batch.policy.loss_mask)
            centering_metrics[f"score_centering/{name}_tail_mass_mean"] = masked_mean(
                tail, batch.policy_data_weights
            ).item()
            if name == "behavior":
                centering_metrics["score_centering/behavior_tail_mass_gt_1pct_fraction"] = masked_mean(
                    (tail > 0.01).float(), batch.policy_data_weights
                ).item()
    mask = batch.policy.loss_mask
    entropy = reduce_to_step(batch.token_entropy, mask, counts.mask, LossReduction.TOKEN_MEAN, **common)
    if config.use_kl_loss:
        if batch.ref_log_probs is None:
            raise ValueError("base_action_log_probs are required when use_kl_loss is enabled")
        kl_values = differentiable_approx_kl(
            batch.policy.log_probs, batch.ref_log_probs, kl_estimator_type=config.kl_estimator_type
        )
        kl = reduce_to_step(kl_values, mask, counts.mask, LossReduction.SEQUENCE_MEAN, **common)
    else:
        kl = policy_row.new_zeros(())
    combined = policy_row + config.kl_loss_coef * kl
    if config.use_entropy_loss:
        combined = combined - config.entropy_loss_coef * entropy
    metrics = complete_clip_metrics(policy.metrics)
    metrics.update(centering_metrics)
    teacher_row = None
    if batch.teacher is not None:
        teacher = batch.teacher
        result = topk_teacher_loss(
            teacher.evidence,
            teacher.student_log_probs_on_support,
            teacher.params,
            vocabulary_size=teacher.vocabulary_size,
        )
        assert result.values.shape == mask.shape
        teacher_row = reduce_to_step(
            result.values,
            mask * teacher.evidence.valid_mask,
            counts.teacher,
            mode,
            numerator_weights=teacher.evidence.loss_weights,
            **common,
        )
        combined = combined + teacher_row
        metrics.update(result.metrics)
        metrics["distillation_loss"] = (teacher_row.detach() * report_scale).item()
    return PolicyObjective(
        optimization_loss=combined * loss_scale,
        rows=ObjectiveRows(
            policy=policy_row * report_scale,
            kl=kl * report_scale,
            entropy=entropy * report_scale,
            teacher=None if teacher_row is None else teacher_row * report_scale,
        ),
        metrics=metrics,
    )


def megatron_loss_scale(num_microbatches: int, data_parallel_size: int) -> float:
    """Cancel the scheduler's microbatch division and the DP gradient average."""
    return float(num_microbatches * data_parallel_size)
