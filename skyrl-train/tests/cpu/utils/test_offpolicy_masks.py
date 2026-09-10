"""Independent numerical and gradient contracts for composable correction masks."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.utils.offpolicy_masks import apply_offpolicy_masks
from skyrl_train.utils.policy_losses import compute_policy_objective, LossScaling
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry, rollout_logprobs_required
from skyrl_train.utils.offpolicy_masks import validate_offpolicy_masks
from tests.offpolicy_mask_reference import offpolicy_keep_reference


def mask_config(**changes):
    config = OmegaConf.create(
        {
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "think_token_weight": 1.0,
            "use_tis": False,
            "tis_imp_ratio_cap": 5.0,
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "max_seq_len": 3,
            "cispo": {"cispo_eps_clip_low": 0.2, "cispo_eps_clip_high": 0.2},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "use_entropy_loss": False,
            "entropy_loss_coef": 0.0,
            "offpolicy_mask": {
                "enabled": False,
                "ratio": "mismatch",
                "low": 0.5,
                "high": 5.0,
                "veto_ratio": 1e-5,
                "renormalize": False,
            },
        }
    )
    for key, value in changes.items():
        OmegaConf.update(config, key, value)
    return config


def transform(config, current, old, rollout, advantages=None, mask=None):
    advantages = torch.ones_like(current) if advantages is None else advantages
    return apply_offpolicy_masks(
        action_log_probs=current,
        old_action_log_probs=old,
        rollout_logprobs=rollout,
        advantages=advantages,
        loss_mask=mask,
        token_entropy=torch.ones_like(current),
        config=config,
    )


def test_disabled_preserves_input_identity_and_stable_zero_schema():
    x = torch.tensor([[-1.0, -1.0, -1.0]])
    advantages = torch.ones_like(x)
    mask = torch.ones_like(x)
    off = transform(mask_config(), x, x, x, advantages, mask)
    on = transform(mask_config(**{"offpolicy_mask.enabled": True}), x, x, x, advantages, mask)
    assert off.advantages is advantages and off.loss_mask is mask
    assert off.metrics.keys() == on.metrics.keys()
    assert set(off.metrics.values()) == {0.0}


def test_icepop_mask_anchor_keeps_denominator_and_composes_with_tis_weights():
    old = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64)
    rollout = torch.tensor([[-0.5, -3.5, -0.8]], dtype=torch.float64)
    result = transform(mask_config(**{"offpolicy_mask.enabled": True}), old, old, rollout)
    assert result.loss_mask is None
    torch.testing.assert_close(result.advantages, torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64))
    assert float(-result.advantages.mean()) == pytest.approx(-2 / 3)
    tis_loss = -(result.advantages * (old - rollout).exp().clamp_max(5)).mean()
    assert float(tis_loss) == pytest.approx(-(math.exp(-0.5) + math.exp(-0.2)) / 3)


def test_sequence_veto_removes_whole_selected_row_below_exponent_clip():
    delta = torch.tensor([[math.log(1e-7), 0.0], [math.log(1e-4), 0.0]], dtype=torch.float64)
    zero = torch.zeros_like(delta)
    result = transform(mask_config(**{"offpolicy_mask.enabled": True}), delta, delta, zero)
    torch.testing.assert_close(result.advantages, torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64))
    assert result.metrics["offpolicy_mask/vetoed_sequence_fraction"] == 0.5


def objective(config, current, old, rollout, advantages, mask, **kwargs):
    return compute_policy_objective(
        action_log_probs=current,
        old_action_log_probs=old,
        base_action_log_probs=None,
        advantages=advantages,
        loss_mask=mask,
        rollout_logprobs=rollout,
        response_span_tags=None,
        token_entropy=torch.ones_like(current),
        config=config,
        policy_loss_fn=PolicyLossRegistry.get(config.policy_loss_type),
        accumulation_steps=1,
        scaling=LossScaling.CALLER,
        **kwargs,
    )


@pytest.mark.parametrize("tis", [False, True])
def test_actual_objective_matches_verl_anchor_with_and_without_tis(tis):
    cfg = mask_config(**{"offpolicy_mask.enabled": True, "use_tis": tis})
    old = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64)
    current = old.clone().requires_grad_()
    rollout = torch.tensor([[-0.5, -3.5, -0.8]], dtype=torch.float64)
    out = objective(cfg, current, old, rollout, torch.ones_like(old), torch.ones_like(old))
    weights = torch.tensor([[math.exp(-0.5), 0.0, math.exp(-0.2)]]) if tis else torch.tensor([[1.0, 0.0, 1.0]])
    assert out.policy_loss.item() == pytest.approx(-weights.mean().item(), abs=1e-7)
    (gradient,) = torch.autograd.grad(out.policy_loss, current)
    torch.testing.assert_close(gradient, -weights.double() / 3, atol=1e-7, rtol=0)


@pytest.mark.parametrize("loss", ["regular", "dual_clip", "behavior_clip", "cispo", "gspo"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_actual_objective_gradient_matches_independent_loss_formula(loss, renormalize):
    cfg = mask_config(
        **{
            "offpolicy_mask.enabled": True,
            "offpolicy_mask.renormalize": renormalize,
            "policy_loss_type": loss,
            "loss_reduction": "sequence_mean" if loss == "gspo" else "token_mean",
        }
    )
    old = torch.full((2, 3), -2.0, dtype=torch.float64)
    delta = torch.tensor([[0.03, 0.12, -0.09], [0.04, -0.10, 0.05]], dtype=torch.float64)
    current = (old + delta).requires_grad_()
    rollout = old - torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=torch.float64)
    mask = torch.ones_like(old)
    out = objective(cfg, current, old, rollout, advantages, mask)
    reference = current.detach().clone().requires_grad_()
    keep = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float64)
    selected = keep if renormalize else mask
    effective_adv = advantages if renormalize else advantages * keep
    if loss == "gspo":
        # GSPO-token uses a detached *selected sequence* ratio, not the old full-row ratio.
        sequence_delta = ((reference - old) * selected).sum(-1, keepdim=True) / selected.sum(-1, keepdim=True)
        ratio = (reference - reference.detach() + sequence_delta.detach()).exp()
    else:
        # Native token-ratio losses evaluate exp in float32, then restore output dtype.
        ratio = (reference - (rollout if loss == "behavior_clip" else old)).float().exp().double()
    if loss == "cispo":
        per_token = -effective_adv * ratio.clamp(0.8, 1.2).detach() * reference
    else:
        per_token = torch.maximum(-effective_adv * ratio, -effective_adv * ratio.clamp(0.8, 1.2))
        if loss in ("dual_clip", "behavior_clip"):
            per_token = torch.where(effective_adv < 0, torch.minimum(per_token, -3 * effective_adv), per_token)
    expected = ((per_token * selected).sum(-1) / selected.sum(-1)).mean()
    (actual_grad,) = torch.autograd.grad(out.policy_loss, current)
    (expected_grad,) = torch.autograd.grad(expected, reference)
    torch.testing.assert_close(out.policy_loss, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=0)
    assert torch.equal(actual_grad[:, 1], torch.zeros(2, dtype=torch.float64))
    if loss == "gspo" and renormalize:
        full_sequence_ratio = delta.mean(-1).exp()
        selected_sequence_ratio = (delta * keep).sum(-1).div(2).exp()
        assert not torch.allclose(full_sequence_ratio, selected_sequence_ratio)


@pytest.mark.parametrize("loss", ["clip_cov", "kl_cov", "sapo"])
def test_validation_rejects_non_advantage_linear_objectives(loss):
    with pytest.raises(ValueError, match="advantage-linear"):
        validate_offpolicy_masks(mask_config(**{"offpolicy_mask.enabled": True, "policy_loss_type": loss}))


def test_renormalization_rejects_external_global_denominator():
    field = "offpolicy_mask"
    cfg = mask_config(**{f"{field}.enabled": True, f"{field}.renormalize": True})
    with pytest.raises(ValueError, match="externally fixed"):
        validate_offpolicy_masks(cfg, global_loss_denom=12)
    cfg.loss_reduction = "seq_mean_token_sum_norm_global"
    with pytest.raises(ValueError, match="externally fixed"):
        validate_offpolicy_masks(cfg)


@pytest.mark.parametrize(
    "changes,required",
    [
        ({}, False),
        ({"offpolicy_mask.enabled": True}, True),
    ],
)
def test_central_rollout_requirement_covers_correction_ratios(changes, required):
    assert rollout_logprobs_required(mask_config(**changes)) is required


def test_full_validate_cfg_rejects_mask_with_kl_cov():
    from tests.cpu.utils.test_policy_optimization import _validatable_dummy_config
    from skyrl_train.utils.utils import validate_cfg

    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.policy_loss_type = "kl_cov"
    cfg.trainer.algorithm.offpolicy_mask = mask_config(**{"offpolicy_mask.enabled": True}).offpolicy_mask
    with pytest.raises(ValueError, match="advantage-linear"):
        validate_cfg(cfg)


@pytest.mark.parametrize("kind", ["mismatch", "full"])
def test_random_band_and_veto_match_independent_scalar_oracle(kind):
    generator = torch.Generator().manual_seed(72)
    old = torch.randn((4, 31), generator=generator, dtype=torch.float64)
    current = old + 0.3
    rollout = old - torch.randn((4, 31), generator=generator, dtype=torch.float64) * 3
    rollout[0, 0] = old[0, 0] + 18
    selected = torch.rand((4, 31), generator=generator) > 0.2
    selected[0, 0] = True
    cfg = mask_config(**{"offpolicy_mask.enabled": True, "offpolicy_mask.ratio": kind})
    result = transform(cfg, current, old, rollout, mask=selected)
    expected = offpolicy_keep_reference((old if kind == "mismatch" else current) - rollout, selected)
    torch.testing.assert_close(result.advantages * selected, expected.double())


def test_disabled_actual_objective_is_exactly_existing_loss():
    cfg = mask_config()
    old = torch.tensor([[-1.0, -2.0, -3.0]])
    current = (old + 0.03).requires_grad_()
    adv = torch.tensor([[1.0, -1.0, 2.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    direct_loss, direct_metrics = PolicyLossRegistry.get("regular")(
        current, old, adv, config=cfg, loss_mask=mask, rollout_logprobs=old
    )
    actual = objective(cfg, current, old, old, adv, mask)
    assert torch.equal(actual.policy_loss, direct_loss)
    assert all(actual.metrics[key] == value for key, value in direct_metrics.items())
    torch.testing.assert_close(
        torch.autograd.grad(actual.policy_loss, current, retain_graph=True)[0],
        torch.autograd.grad(direct_loss, current)[0],
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"offpolicy_mask.ratio": "stale"},
        {"offpolicy_mask.low": 0.0},
        {"offpolicy_mask.high": float("inf")},
        {"use_tis": True, "policy_loss_type": "behavior_clip"},
    ],
)
def test_invalid_transform_config_fails_closed(changes):
    with pytest.raises(ValueError):
        validate_offpolicy_masks(mask_config(**{"offpolicy_mask.enabled": True, **changes}))
