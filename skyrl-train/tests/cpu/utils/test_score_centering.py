"""Independent small-vocabulary checks for PPO/TIS score centering."""

import torch
from omegaconf import OmegaConf

from skyrl_train.utils.policy_losses import LossScaling, compute_policy_objective, ppo_policy_loss
from skyrl_train.utils.score_centering import ppo_tis_score_centering_correction


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


def test_policy_objective_centers_expected_constant_advantage_gradient():
    old = torch.tensor([0.20, 0.25, 0.30, 0.25], dtype=torch.float64)
    behavior = torch.tensor([0.45, 0.08, 0.07, 0.40], dtype=torch.float64)
    logits = torch.tensor([-0.9, 0.8, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
    current = logits.log_softmax(dim=-1)
    config = OmegaConf.create(
        {
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 1,
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "think_token_weight": 1.0,
            "use_entropy_loss": False,
            "entropy_loss_coef": 0.0,
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "use_tis": True,
            "tis_imp_ratio_cap": 1.5,
            "score_centering_topk": 4,
        }
    )
    expected_loss = logits.new_zeros(())
    for sampled_id, weight in enumerate(behavior):
        objective = compute_policy_objective(
            action_log_probs=current[sampled_id].reshape(1, 1),
            old_action_log_probs=old[sampled_id].log().reshape(1, 1),
            base_action_log_probs=None,
            advantages=torch.tensor([[2.3]], dtype=torch.float64),
            loss_mask=torch.ones((1, 1), dtype=torch.float64),
            rollout_logprobs=behavior[sampled_id].log().reshape(1, 1),
            response_span_tags=None,
            token_entropy=torch.zeros((1, 1), dtype=torch.float64),
            config=config,
            policy_loss_fn=ppo_policy_loss,
            accumulation_steps=1,
            scaling=LossScaling.CALLER,
            score_current_topk_logprobs=current.reshape(1, 1, -1),
            score_old_topk_logprobs=old.log().reshape(1, 1, -1),
            score_behavior_topk_logprobs=behavior.log().reshape(1, 1, -1),
        )
        expected_loss = expected_loss + weight * objective.optimization_loss
    gradient = torch.autograd.grad(expected_loss, logits)[0]
    # The production PPO ratio uses float32 exponentiation even for double inputs.
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0)
