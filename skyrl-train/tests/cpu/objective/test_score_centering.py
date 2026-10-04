"""Independent small-vocabulary checks for PPO/TIS score centering."""

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.objective.correction import compute_correction
from skyrl_train.config.objective_spec import LossReduction, off_policy_correction
from skyrl_train.objective.losses import ppo_policy_loss
from skyrl_train.objective.objective import (
    ScoreCenteringBatch,
    build_objective_micro_batch,
    compute_policy_objective,
    megatron_loss_scale,
)
from skyrl_train.objective.reduction import policy_data_weights, step_counts
from skyrl_train.objective.score_centering import ppo_tis_score_centering_correction


def _expected_ppo_tis_loss(logits, old_probs, behavior_probs, advantage, cap, clip_low, clip_high):
    """Enumerate sampled actions under the full behavior distribution."""
    current_logprobs = logits.log_softmax(dim=-1)
    old_logprobs = old_probs.log()
    behavior_logprobs = behavior_probs.log()
    ratio = (current_logprobs - old_logprobs).exp()
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    sampled_loss = -torch.minimum(ratio * advantage, clipped * advantage)
    sampled_loss *= (old_logprobs - behavior_logprobs).exp().clamp(max=cap)
    return (behavior_probs * sampled_loss).sum()


def _correction(logits, old_probs, behavior_probs, advantage, head, cap, clip_low, clip_high):
    current_logprobs = logits.log_softmax(dim=-1)[head].reshape(1, 1, -1)
    old_logprobs = old_probs.log()[head].reshape(1, 1, -1)
    behavior_logprobs = behavior_probs.log()[head].reshape(1, 1, -1)
    return ppo_tis_score_centering_correction(
        current_logprobs,
        old_logprobs,
        behavior_logprobs,
        torch.tensor([[advantage]], dtype=logits.dtype),
        torch.ones((1, 1), dtype=logits.dtype),
        tis_cap=cap,
        eps_clip_low=clip_low,
        eps_clip_high=clip_high,
    ).sum()


def test_full_vocabulary_centering_cancels_constant_reward_gradient_with_clipping():
    old = torch.tensor([0.20, 0.25, 0.30, 0.25], dtype=torch.float64)
    behavior = torch.tensor([0.45, 0.08, 0.07, 0.40], dtype=torch.float64)
    head = torch.arange(4)
    for advantage in (-1.7, 2.3):
        logits = torch.tensor([-0.9, 0.8, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
        loss = _expected_ppo_tis_loss(logits, old, behavior, advantage, 1.5, 0.2, 0.2)
        loss += _correction(logits, old, behavior, advantage, head, 1.5, 0.2, 0.2)
        gradient = torch.autograd.grad(loss, logits)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


def test_unclipped_untruncated_tis_centering_has_zero_gradient():
    old = torch.tensor([0.18, 0.42, 0.14, 0.26], dtype=torch.float64)
    behavior = torch.tensor([0.38, 0.18, 0.24, 0.20], dtype=torch.float64)
    logits = old.log().clone().requires_grad_()
    correction = _correction(logits, old, behavior, 1.0, torch.arange(4), 10.0, 0.5, 0.5)
    gradient = torch.autograd.grad(correction, logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


def test_topk_tail_model_matches_full_gradient_when_tail_ratios_are_constant():
    current = torch.tensor([0.45, 0.30, 0.10, 0.09, 0.06], dtype=torch.float64)
    old = torch.tensor([0.33, 0.27, 0.16, 0.144, 0.096], dtype=torch.float64)
    behavior = torch.tensor([0.60, 0.16, 0.096, 0.0864, 0.0576], dtype=torch.float64)
    # The tail of both fixed policies is proportional to current on IDs 2:.
    head = torch.tensor([0, 1])
    logits = current.log().clone().requires_grad_()
    topk_gradient = torch.autograd.grad(_correction(logits, old, behavior, -1.0, head, 1.3, 0.2, 0.2), logits)[0]
    full_gradient = torch.autograd.grad(
        _correction(logits, old, behavior, -1.0, torch.arange(5), 1.3, 0.2, 0.2), logits
    )[0]
    torch.testing.assert_close(topk_gradient, full_gradient, atol=1e-12, rtol=0)


def test_float32_subfloor_tails_track_full_vocabulary_score_gradient():
    # Both omitted masses are below the production 1e-6 floor, and they differ.
    current_probs = torch.tensor([0.55 * (1 - 8e-7), 0.45 * (1 - 8e-7), 8e-7], dtype=torch.float32)
    behavior_probs = torch.tensor([0.45 * (1 - 2e-7), 0.55 * (1 - 2e-7), 2e-7], dtype=torch.float32)
    logits = current_probs.log().clone().requires_grad_()
    current_logprobs = logits.log_softmax(dim=-1)
    assert 0 < 1 - current_logprobs[:2].detach().exp().sum() < 1e-6
    assert 0 < 1 - behavior_probs[:2].sum() < 1e-6

    correction = ppo_tis_score_centering_correction(
        current_logprobs[:2].reshape(1, 1, 2),
        current_probs[:2].log().reshape(1, 1, 2),
        behavior_probs[:2].log().reshape(1, 1, 2),
        torch.ones((1, 1), dtype=torch.float32),
        torch.ones((1, 1), dtype=torch.float32),
        tis_cap=1.05,
        eps_clip_low=0.2,
        eps_clip_high=0.2,
    )
    actual = torch.autograd.grad(correction.sum(), logits)[0]
    p = current_logprobs.detach().exp()
    full_coefficient = torch.minimum(p, 1.05 * behavior_probs)
    expected = full_coefficient - p * full_coefficient.sum()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_masked_sentinel_rows_do_not_poison_centering():
    logits = torch.tensor([[[[-0.3, 0.1, 0.2]]]], dtype=torch.float64, requires_grad=True)
    selected = logits.log_softmax(dim=-1)[..., :2].squeeze(0)
    invalid = torch.full_like(selected, torch.nan)
    values = torch.cat((selected, invalid), dim=1)
    loss = ppo_tis_score_centering_correction(
        values,
        values.detach(),
        values.detach(),
        torch.ones((1, 2), dtype=torch.float64),
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        tis_cap=2.0,
        eps_clip_low=0.2,
        eps_clip_high=0.2,
    )
    assert torch.isfinite(loss).all()
    assert loss[0, 1] == 0


@pytest.mark.parametrize(("enabled", "capture_width"), [(False, 0), (False, 4), (True, 4)])
@pytest.mark.parametrize("mode", list(LossReduction))
@pytest.mark.parametrize(("micro_size", "dp"), [(4, 1), (1, 1), (2, 2), (1, 2)])
def test_composed_ppo_tis_centering_matches_enumerated_value_gradient_and_partition(
    enabled, capture_width, mode, micro_size, dp
):
    config = OmegaConf.create(
        dict(
            policy_loss_type="regular",
            loss_reduction=mode.value,
            eps_clip_low=0.2,
            eps_clip_high=0.2,
            off_policy_correction="custom",
            off_policy_correction_rules=[dict(kind="token", action="truncate", high=1.05)],
            use_kl_loss=False,
            kl_loss_coef=0,
            use_entropy_loss=False,
            score_centering_topk=capture_width,
            score_centering_enabled=enabled,
        )
    )
    mask = torch.tensor([[1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]], dtype=torch.float64)
    tags = torch.tensor([[1, 0, 0], [1, 0, 1], [0, 0, 0], [0, 1, 0]])
    weights = policy_data_weights(mask, tags, 0.25)
    advantages = torch.tensor([[2, 0, 0], [-1, 0.5, 3], [0, 0, 0], [-2, 1, 0]], dtype=torch.float64)
    old = torch.tensor([0.20, 0.25, 0.30, 0.15, 0.10], dtype=torch.float64).expand(4, 3, 5)
    behavior = torch.tensor([0.45, 0.08, 0.07, 0.30, 0.10], dtype=torch.float64).expand(4, 3, 5)
    # These actions exercise positive/negative advantages and active/inactive PPO clipping.
    actions = torch.tensor([[1, 0, 0], [0, 2, 4], [0, 0, 0], [3, 2, 0]])
    logits = torch.tensor([-0.9, 0.8, -0.2, 0.1, -1.0], dtype=torch.float64).expand(4, 3, 5).clone()
    logits.requires_grad_()
    current = logits.log_softmax(-1)
    selected = current.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    old_selected = old.log().gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    behavior_selected = behavior.log().gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    correction = compute_correction(old_selected, behavior_selected, mask, off_policy_correction(config)).weights
    counts = step_counts([weights], [mask], [], [advantages], 8, lambda value: value)

    # Enumerate the intended formula independently. The SC expectation includes TIS
    # once, while the sampled PPO term receives the sampled action's TIS weight.
    expected = logits.new_zeros(())
    for row in range(4):
        for token in range(3):
            if not mask[row, token]:
                continue
            action = actions[row, token]
            ratio = (current[row, token, action] - old[row, token, action].log()).float().exp().double()
            advantage = advantages[row, token]
            tis = min(float(old[row, token, action] / behavior[row, token, action]), 1.05)
            term = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage) * tis
            if enabled:
                # The one omitted action is an exact tail. Eliminate its score with
                # sum_v p_v grad(log p_v)=0, independently of the production helper.
                tail_p = current[row, token, 4].detach().exp()
                tail_o, tail_q = old[row, token, 4], behavior[row, token, 4]
                tail_active = bool(tail_p / tail_o <= 1.2 if advantage >= 0 else tail_p / tail_o >= 0.8)
                tail_coefficient = tail_q / tail_o * min(float(tail_o / tail_q), 1.05) * tail_active
                for candidate in range(4):
                    p = current[row, token, candidate].detach().exp()
                    o, q = old[row, token, candidate], behavior[row, token, candidate]
                    active = bool(p / o <= 1.2 if advantage >= 0 else p / o >= 0.8)
                    coefficient = q * min(float(o / q), 1.05) * p / o * active
                    term = term + advantage * (coefficient - tail_coefficient * p) * current[row, token, candidate]
            term = term * weights[row, token]
            if mode == LossReduction.SEQUENCE_MEAN:
                term = term / weights[row].sum()
            expected = expected + term
    denominator = weights.sum() if mode == LossReduction.TOKEN_MEAN else (weights.sum(-1) > 0).sum()
    if mode in (LossReduction.SEQ_MEAN_TOKEN_SUM_NORM, LossReduction.SEQ_MEAN_TOKEN_SUM_NORM_GLOBAL):
        denominator = denominator * 8
    expected = expected / denominator
    expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]

    micros = 4 // dp // micro_size
    scale = megatron_loss_scale(micros, dp)
    actual = logits.new_zeros(())
    reported = logits.new_zeros(())
    for start in range(0, 4, micro_size):
        chunk = slice(start, start + micro_size)
        evidence = None
        if capture_width:
            invalid = ~mask[chunk].bool().unsqueeze(-1)
            evidence = ScoreCenteringBatch(
                current[chunk, :, :4].masked_fill(invalid, torch.nan),
                old[chunk, :, :4].log().masked_fill(invalid, torch.nan),
                behavior[chunk, :, :4].log().masked_fill(invalid, torch.nan),
            )
        batch = build_objective_micro_batch(
            action_log_probs=selected[chunk].masked_fill(~mask[chunk].bool(), torch.nan),
            old_action_log_probs=old_selected[chunk],
            base_action_log_probs=None,
            advantages=advantages[chunk],
            loss_mask=mask[chunk],
            rollout_logprobs=behavior_selected[chunk],
            response_span_tags=tags[chunk],
            token_entropy=torch.zeros_like(mask[chunk]),
            think_token_weight=0.25,
            teacher=None,
            correction_weights=correction[chunk],
            score_centering=evidence,
        )
        objective = compute_policy_objective(
            batch,
            loss=ppo_policy_loss,
            counts=counts,
            config=config,
            loss_scale=scale,
            report_scale=scale,
        )
        actual = actual + objective.optimization_loss / micros / dp
        reported = reported + objective.rows.policy / scale
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(reported, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-7, rtol=1e-6)
    assert torch.isfinite(actual_gradient).all()


@pytest.mark.parametrize("advantage", [-1.7, 2.3])
def test_composed_full_vocabulary_centering_cancels_expected_constant_advantage_gradient(advantage):
    config = OmegaConf.create(
        dict(
            loss_reduction="token_mean",
            eps_clip_low=0.2,
            eps_clip_high=0.2,
            off_policy_correction="custom",
            off_policy_correction_rules=[dict(kind="token", action="truncate", high=1.5)],
            use_kl_loss=False,
            kl_loss_coef=0,
            use_entropy_loss=False,
            score_centering_topk=4,
        )
    )
    old = torch.tensor([0.20, 0.25, 0.30, 0.25], dtype=torch.float64)
    behavior = torch.tensor([0.45, 0.08, 0.07, 0.40], dtype=torch.float64)
    logits = torch.tensor([-0.9, 0.8, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
    current = logits.log_softmax(-1)
    mask = torch.ones((1, 1), dtype=torch.float64)
    advantages = torch.full_like(mask, advantage)
    counts = step_counts([mask], [mask], [], [advantages], 1, lambda value: value)
    expected = logits.new_zeros(())
    for action, probability in enumerate(behavior):
        old_selected = old[action].log().reshape(1, 1)
        behavior_selected = behavior[action].log().reshape(1, 1)
        correction = compute_correction(old_selected, behavior_selected, mask, off_policy_correction(config)).weights
        batch = build_objective_micro_batch(
            action_log_probs=current[action].reshape(1, 1),
            old_action_log_probs=old_selected,
            base_action_log_probs=None,
            advantages=advantages,
            loss_mask=mask,
            rollout_logprobs=behavior_selected,
            response_span_tags=None,
            token_entropy=torch.zeros_like(mask),
            think_token_weight=1,
            teacher=None,
            correction_weights=correction,
            score_centering=ScoreCenteringBatch(
                current.reshape(1, 1, 4),
                old.log().reshape(1, 1, 4),
                behavior.log().reshape(1, 1, 4),
            ),
        )
        objective = compute_policy_objective(
            batch, loss=ppo_policy_loss, counts=counts, config=config, loss_scale=1, report_scale=1
        )
        expected = expected + probability * objective.optimization_loss
    gradient = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0)
