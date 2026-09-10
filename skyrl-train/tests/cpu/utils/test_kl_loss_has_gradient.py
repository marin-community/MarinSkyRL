"""The KL loss must carry a gradient: compute_approx_kl is @torch.no_grad (metrics),
differentiable_approx_kl is not (loss)."""

import torch
from omegaconf import OmegaConf

from skyrl_train.utils.policy_math import compute_approx_kl, differentiable_approx_kl


def _grad_norm(coef: float, estimator: str = "k3") -> float:
    torch.manual_seed(0)
    logp = torch.randn(2, 6, requires_grad=True)
    base = (logp.detach() + 0.3 * torch.randn(2, 6)).detach()
    mask = torch.ones(2, 6)
    kl = differentiable_approx_kl(logp, base, loss_mask=mask, kl_estimator_type=estimator)
    loss = (kl * mask).sum() / mask.sum() * coef
    loss.backward()
    return float(logp.grad.norm())


def test_differentiable_approx_kl_is_differentiable_and_scales_with_coef():
    g1 = _grad_norm(1.0)
    g10 = _grad_norm(10.0)
    assert g1 > 0.0
    assert abs(g10 - 10.0 * g1) < 1e-4 * max(1.0, g10)


def test_compute_approx_kl_is_metrics_only():
    logp = torch.randn(2, 6, requires_grad=True)
    base = torch.randn(2, 6)
    kl = compute_approx_kl(logp, base, loss_mask=torch.ones(2, 6))
    assert not kl.requires_grad
    assert torch.allclose(kl, differentiable_approx_kl(logp, base, loss_mask=torch.ones(2, 6)).detach())


def test_policy_loss_kl_term_has_gradient():
    from skyrl_train.utils.policy_losses import _compute_policy_auxiliary_terms

    cfg = OmegaConf.create(
        {
            "use_kl_loss": True,
            "kl_loss_coef": 0.1,
            "kl_estimator_type": "k3",
            "use_entropy_loss": False,
            "entropy_loss_coef": 0.0,
        }
    )
    logp = torch.randn(2, 6, requires_grad=True)
    base = torch.randn(2, 6)
    terms = _compute_policy_auxiliary_terms(
        action_log_probs=logp,
        base_action_log_probs=base,
        token_entropy=torch.zeros(2, 6),
        loss_mask=torch.ones(2, 6),
        config=cfg,
    )
    terms.loss.backward()
    assert logp.grad is not None and float(logp.grad.abs().sum()) > 0.0
