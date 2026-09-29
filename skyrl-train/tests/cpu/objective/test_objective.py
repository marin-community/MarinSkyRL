import pytest
import torch
from omegaconf import OmegaConf

from marinskyrl.distillation import DistillationObjectiveKind
from skyrl_train.config.objective_spec import LossReduction, TopKLossParams
from skyrl_train.distillation import TeacherTopKInput
from skyrl_train.objective.losses import behavior_clipped_policy_loss, importance_sampling_policy_loss, ppo_policy_loss
from skyrl_train.objective.objective import (
    TopKTeacherBatch,
    build_objective_micro_batch,
    compute_policy_objective,
    megatron_loss_scale,
)
from skyrl_train.objective.reduction import policy_data_weights, step_counts


@pytest.mark.parametrize(
    "mode", [LossReduction.TOKEN_MEAN, LossReduction.SEQUENCE_MEAN, LossReduction.SEQ_MEAN_TOKEN_SUM_NORM]
)
@pytest.mark.parametrize(("micro_size", "dp"), [(1, 1), (2, 1), (4, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("backend", ["cpu", "megatron"])
@pytest.mark.parametrize("reward_mode", ["add", "replace"])
def test_composed_rows_match_full_batch_value_gradient_and_reporting(mode, micro_size, dp, backend, reward_mode):
    config = OmegaConf.create(
        dict(
            policy_loss_type="behavior_clip" if reward_mode == "replace" else "importance_sampling",
            loss_reduction=mode.value,
            use_kl_loss=True,
            kl_loss_coef=0.7,
            kl_estimator_type="k2",
            use_entropy_loss=True,
            entropy_loss_coef=0.3,
            distillation=dict(reward_mode=reward_mode),
        )
    )
    mask = torch.tensor([[1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]], dtype=torch.float64)
    tags = torch.tensor([[1, 0, 0], [1, 0, 1], [0, 0, 0], [0, 1, 0]])
    teacher_mask = mask.bool() & torch.tensor([[1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 0, 0]], dtype=torch.bool)
    if reward_mode == "replace":
        mask = mask * teacher_mask
    weights = policy_data_weights(mask, tags, 0.25)
    correction = torch.tensor([[0.5, 0, 0], [0, 2, 1.5], [0, 0, 0], [0.75, 1, 0]])
    route_weights = torch.tensor([[0.2, 0, 0], [0, 2, 0.5], [0, 0, 0], [1.5, 0, 0]])
    advantages = torch.tensor([[2, 99, 99], [-1, 0.5, 3], [99, 99, 99], [0, 2, 99]])
    log_probs = torch.where(mask.bool(), torch.full_like(mask, -0.8), torch.nan).requires_grad_()
    entropy = torch.where(mask.bool(), torch.full_like(mask, 0.9), torch.nan).requires_grad_()
    support = torch.tensor([0.25, 0.35], dtype=torch.float64).log().expand(4, 3, 2).clone()
    support[~teacher_mask] = torch.nan
    support.requires_grad_()
    old = torch.full_like(mask, -1.0)
    reference = torch.full_like(mask, -1.3)
    counts = step_counts([weights], [mask], [teacher_mask], [advantages], 8, lambda value: value)

    # Enumerate response positions independently of production reduction and teacher math.
    expected = {name: log_probs.new_zeros(()) for name in ("policy", "kl", "entropy", "teacher")}
    data = {"policy": weights, "teacher": teacher_mask.double()}
    for i in range(4):
        for j in range(3):
            if mask[i, j]:
                term = -(log_probs[i, j] - old[i, j]).exp() * advantages[i, j] * weights[i, j] * correction[i, j]
                if mode == LossReduction.SEQUENCE_MEAN:
                    term = term / weights[i].sum()
                expected["policy"] = expected["policy"] + term * (reward_mode == "add")
                expected["kl"] = expected["kl"] + 0.5 * (log_probs[i, j] - reference[i, j]).square() / mask[i].sum() / 3
                expected["entropy"] = expected["entropy"] + entropy[i, j] / mask.sum()
            if teacher_mask[i, j]:
                term = sum(q * (torch.tensor(q).log() - support[i, j, k]) for k, q in enumerate((0.6, 0.4)))
                term = term * route_weights[i, j]
                if mode == LossReduction.SEQUENCE_MEAN:
                    term = term / teacher_mask[i].sum()
                expected["teacher"] = expected["teacher"] + term
    for row in ("policy", "teacher"):
        denominator = data[row].sum() if mode == LossReduction.TOKEN_MEAN else (data[row].sum(-1) > 0).sum()
        if mode == LossReduction.SEQ_MEAN_TOKEN_SUM_NORM:
            denominator = denominator * 8
        expected[row] = expected[row] / denominator
    expected_loss = expected["policy"] + 0.7 * expected["kl"] - 0.3 * expected["entropy"] + expected["teacher"]
    expected_grads = torch.autograd.grad(expected_loss, (log_probs, entropy, support), retain_graph=True)

    micros = 4 // dp // micro_size
    report_scale = micros * dp
    loss_scale = megatron_loss_scale(micros, dp) if backend == "megatron" else dp
    actual_loss = log_probs.new_zeros(())
    reported = {name: log_probs.new_zeros(()) for name in expected}
    for start in range(0, 4, micro_size):
        chunk = slice(start, start + micro_size)
        valid = teacher_mask[chunk]
        evidence = TeacherTopKInput(
            torch.tensor([0, 1]).expand(micro_size, 3, 2),
            torch.tensor([0.42, 0.28]).log().expand(micro_size, 3, 2),
            torch.full_like(mask[chunk], 0.7),
            valid,
            route_weights[chunk],
        )
        batch = build_objective_micro_batch(
            action_log_probs=log_probs[chunk],
            old_action_log_probs=old[chunk],
            base_action_log_probs=reference[chunk],
            advantages=advantages[chunk],
            loss_mask=mask[chunk],
            rollout_logprobs=None,
            correction_weights=correction[chunk],
            response_span_tags=tags[chunk],
            token_entropy=entropy[chunk],
            think_token_weight=0.25,
            teacher=TopKTeacherBatch(
                evidence, support[chunk], TopKLossParams(DistillationObjectiveKind.SPARSE_FORWARD_KL, 0.2, 0.2, 3), 3
            ),
        )
        objective = compute_policy_objective(
            batch,
            loss=behavior_clipped_policy_loss if reward_mode == "replace" else importance_sampling_policy_loss,
            counts=counts,
            config=config,
            loss_scale=loss_scale,
            report_scale=report_scale,
        )
        actual_loss = actual_loss + objective.optimization_loss / dp / (micros if backend == "megatron" else 1)
        for name in reported:
            reported[name] = reported[name] + getattr(objective.rows, name) / report_scale
    actual_grads = torch.autograd.grad(actual_loss, (log_probs, entropy, support))
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-7)
    for name in reported:
        torch.testing.assert_close(reported[name], expected[name], rtol=1e-6, atol=1e-7)
    for actual, target in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, target, rtol=1e-6, atol=1e-7)
        assert torch.isfinite(actual).all()


@pytest.mark.parametrize(("micro_batch", "dp_size"), [(1, 1), (2, 1), (4, 1), (1, 2), (2, 2)])
def test_step_objective_matches_full_batch_for_any_split(micro_batch, dp_size):
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float64)
    advantages = torch.arange(1, 17, dtype=torch.float64).reshape(4, 4)
    reference_log_probs = torch.zeros_like(advantages, requires_grad=True)
    reference = (
        sum(
            -reference_log_probs[row, token].exp() * advantages[row, token]
            for row in range(4)
            for token in range(4)
            if mask[row, token] > 0
        )
        / mask.sum()
    )
    reference.backward()

    log_probs = torch.zeros_like(advantages, requires_grad=True)
    config = OmegaConf.create(
        {
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "policy_loss_type": "regular",
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "think_token_weight": 1.0,
            "use_entropy_loss": False,
            "entropy_loss_coef": 0.0,
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "kl_estimator_type": "k1",
        }
    )
    counts = step_counts([mask], [mask], [], [advantages], 4, lambda value: value)
    rank_rows = 4 // dp_size
    num_microbatches = rank_rows // micro_batch
    cp_size = 1
    scheduled = log_probs.new_zeros(())
    for rank in range(dp_size):
        for start in range(rank * rank_rows, (rank + 1) * rank_rows, micro_batch):
            rows = slice(start, start + micro_batch)
            batch = build_objective_micro_batch(
                action_log_probs=log_probs[rows],
                old_action_log_probs=torch.zeros_like(log_probs[rows]),
                base_action_log_probs=None,
                advantages=advantages[rows],
                loss_mask=mask[rows],
                rollout_logprobs=None,
                response_span_tags=None,
                token_entropy=torch.zeros_like(log_probs[rows]),
                think_token_weight=1.0,
                teacher=None,
            )
            objective = compute_policy_objective(
                batch,
                loss=ppo_policy_loss,
                counts=counts,
                config=config,
                loss_scale=megatron_loss_scale(num_microbatches, dp_size),
                report_scale=num_microbatches * dp_size,
            )
            # Megatron's schedule scales by CP/M and DDP averages over DP*CP.
            scheduled = scheduled + objective.optimization_loss * cp_size / num_microbatches / (dp_size * cp_size)
    scheduled.backward()
    torch.testing.assert_close(log_probs.grad, reference_log_probs.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(scheduled, reference, rtol=1e-6, atol=1e-7)
