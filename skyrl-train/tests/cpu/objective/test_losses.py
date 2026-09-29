import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.objective.losses import (
    PolicyLossInputs,
    behavior_clipped_policy_loss,
    compute_policy_loss_cispo,
    compute_policy_loss_clip_cov,
    compute_policy_loss_kl_cov,
    dual_clip_policy_loss,
    gspo_policy_loss,
    importance_sampling_policy_loss,
    ppo_policy_loss,
    sapo_policy_loss,
    sft_policy_loss,
)


LOSSES = [
    ppo_policy_loss,
    dual_clip_policy_loss,
    gspo_policy_loss,
    compute_policy_loss_cispo,
    sapo_policy_loss,
    compute_policy_loss_clip_cov,
    compute_policy_loss_kl_cov,
    importance_sampling_policy_loss,
    behavior_clipped_policy_loss,
    sft_policy_loss,
]


@pytest.fixture
def loss_config():
    return OmegaConf.create(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "cispo": {"cispo_eps_clip_low": 1.0, "cispo_eps_clip_high": 0.2},
            "sapo": {"tau_pos": 1.0, "tau_neg": 1.05},
            "clip_cov": {"clip_ratio": 0.2, "clip_cov_lb": 1.0, "clip_cov_ub": 5.0},
            "kl_cov": {"kl_cov_frac": 0.2, "ppo_kl_coef": 0.1},
        }
    )


@pytest.mark.parametrize(
    "loss", [loss for loss in LOSSES if loss not in (sft_policy_loss, compute_policy_loss_clip_cov)]
)
def test_policy_loss_on_policy_gradient_equals_policy_gradient(loss, loss_config):
    log_probs = torch.tensor([[-0.2, -0.6, -0.8], [-1.5, -0.4, 0]], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([[2.0, 2.0, 2.0], [-1.0, -1.0, 0]], dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float64)
    inputs = PolicyLossInputs(log_probs, log_probs.detach(), log_probs.detach(), advantages, mask)
    values = loss(inputs, loss_config).values
    reduction = (values.sum(-1) / mask.sum(-1)).mean()
    actual = torch.autograd.grad(reduction, log_probs)[0]
    reference = (-(advantages * log_probs * mask).sum(-1) / mask.sum(-1)).mean()
    expected = torch.autograd.grad(reference, log_probs)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("loss", LOSSES)
def test_policy_loss_is_finite_at_extreme_ratios(loss, loss_config):
    log_probs = torch.tensor([[100.0, -100.0]], requires_grad=True)
    inputs = PolicyLossInputs(
        log_probs,
        torch.zeros_like(log_probs),
        torch.zeros_like(log_probs),
        torch.tensor([[1.0, -1.0]]),
        torch.ones_like(log_probs),
    )
    result = loss(inputs, loss_config)
    result.values.sum().backward()
    assert torch.isfinite(result.values).all()
    assert torch.isfinite(log_probs.grad).all()


@pytest.mark.parametrize("loss", [loss for loss in LOSSES if loss not in (sft_policy_loss, compute_policy_loss_kl_cov)])
def test_advantage_linear_losses_vanish_at_zero_advantage(loss, loss_config):
    log_probs = torch.tensor([[0.5, -3.0, 0.0]], requires_grad=True)
    inputs = PolicyLossInputs(
        log_probs,
        torch.zeros_like(log_probs),
        torch.zeros_like(log_probs),
        torch.zeros_like(log_probs),
        torch.tensor([[1.0, 1.0, 0.0]]),
    )
    result = loss(inputs, loss_config)
    result.values.sum().backward()
    torch.testing.assert_close(result.values, torch.zeros_like(log_probs))
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs))
