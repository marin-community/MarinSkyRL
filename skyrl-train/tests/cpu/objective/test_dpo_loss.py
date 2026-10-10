import math

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from skyrl_train.objective.dpo import DPOInputs, dpo_pair_values
from skyrl_train.objective.losses import PolicyLossInputs, dpo_policy_loss


def make_inputs(seed=0, rows=4, cols=3, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    log_probs = torch.randn(rows, cols, generator=generator, dtype=dtype, requires_grad=True)
    ref = torch.randn(rows, cols, generator=generator, dtype=dtype)
    mask = torch.ones(rows, cols, dtype=dtype)
    roles = torch.tensor([1.0, -1.0], dtype=dtype).repeat(rows // 2)
    return log_probs, ref, mask, roles


def reference_loss(log_probs, ref, mask, beta, smoothing):
    """TRL's sigmoid/robust pair formula computed pair-wise."""
    pairs = log_probs.reshape(-1, 2, log_probs.shape[-1])
    ref_pairs = ref.reshape(-1, 2, ref.shape[-1])
    deltas = ((pairs - ref_pairs) * mask.reshape(-1, 2, mask.shape[-1])).sum(-1)
    delta = deltas[:, 0] - deltas[:, 1]
    scale = 1.0 / (1.0 - 2.0 * smoothing)
    loss = scale * (-(1 - smoothing) * F.logsigmoid(beta * delta) - smoothing * F.logsigmoid(-beta * delta))
    return loss.mean(), delta


def pair_values(log_probs, ref, mask, roles, beta, smoothing):
    return dpo_pair_values(log_probs, ref, mask, DPOInputs(roles), beta, smoothing)


@pytest.mark.parametrize("beta", [0.05, 0.1, 1.0])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_dpo_gradient_matches_reference_formula(beta, smoothing):
    log_probs, ref, mask, roles = make_inputs()
    values, _ = pair_values(log_probs, ref, mask, roles, beta, smoothing)
    surrogate = values.sum() / (log_probs.shape[0] // 2)
    actual = torch.autograd.grad(surrogate, log_probs)[0]
    expected_loss, _ = reference_loss(log_probs, ref, mask, beta, smoothing)
    expected = torch.autograd.grad(expected_loss, log_probs)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("smoothing", [0.0, 0.2])
def test_dpo_metrics_match_closed_forms(smoothing):
    beta = 0.1
    log_probs, ref, mask, roles = make_inputs(seed=3)
    _values, metrics = pair_values(log_probs, ref, mask, roles, beta, smoothing)
    expected_loss, delta = reference_loss(log_probs, ref, mask, beta, smoothing)
    chosen_reward = beta * (log_probs[0::2] - ref[0::2]).sum(-1)
    rejected_reward = beta * (log_probs[1::2] - ref[1::2]).sum(-1)
    assert metrics["dpo/loss"] == pytest.approx(float(expected_loss))
    assert metrics["dpo/accuracy"] == pytest.approx(float((delta > 0).float().mean()))
    assert metrics["dpo/margin"] == pytest.approx(float((chosen_reward - rejected_reward).mean()))
    assert metrics["dpo/chosen_reward"] == pytest.approx(float(chosen_reward.mean()))
    assert metrics["dpo/rejected_reward"] == pytest.approx(float(rejected_reward.mean()))
    if smoothing == 0.0:
        # At the reference policy the loss is log 2 and the surrogate row is zero.
        zero_values, zero_metrics = pair_values(ref, ref, mask, roles, beta, 0.0)
        assert zero_metrics["dpo/loss"] == pytest.approx(math.log(2.0))
        assert zero_values.sum().item() == pytest.approx(0.0, abs=1e-9)


def test_registered_loss_reads_config_and_masks_values():
    log_probs, ref, mask, roles = make_inputs()
    config = OmegaConf.create({"dpo": {"beta": 0.3, "label_smoothing": 0.0}})
    direct_values, _ = pair_values(log_probs, ref, mask, roles, 0.3, 0.0)
    inputs = PolicyLossInputs(
        log_probs,
        log_probs.detach(),
        None,
        torch.zeros_like(mask),
        mask,
        ref_log_probs=ref,
        dpo=DPOInputs(roles),
    )
    registered = dpo_policy_loss(inputs, config)
    torch.testing.assert_close(registered.values, direct_values)
    # Positions the loss mask excludes carry zero value and zero gradient.
    masked = mask.clone()
    masked[:, -1] = 0
    masked_inputs = PolicyLossInputs(
        log_probs, log_probs.detach(), None, torch.zeros_like(mask), masked, ref_log_probs=ref, dpo=DPOInputs(roles)
    )
    token_loss = dpo_policy_loss(masked_inputs, config)
    assert torch.equal(token_loss.values[:, -1], torch.zeros_like(token_loss.values[:, -1]))


def test_dpo_rejects_broken_pairs():
    log_probs, ref, mask, roles = make_inputs()

    with pytest.raises(ValueError, match="reference"):
        dpo_policy_loss(
            PolicyLossInputs(log_probs, log_probs.detach(), None, torch.zeros_like(mask), mask, dpo=DPOInputs(roles)),
            OmegaConf.create({"dpo": {"beta": 0.1, "label_smoothing": 0.0}}),
        )
    with pytest.raises(ValueError, match="pair_roles"):
        dpo_policy_loss(
            PolicyLossInputs(log_probs, log_probs.detach(), None, torch.zeros_like(mask), mask, ref_log_probs=ref),
            OmegaConf.create({"dpo": {"beta": 0.1, "label_smoothing": 0.0}}),
        )
    with pytest.raises(ValueError, match="even microbatch"):
        pair_values(log_probs[:3], ref[:3], mask[:3], roles[:3], 0.1, 0.0)
    flipped = torch.tensor([-1.0, 1.0, 1.0, -1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="alternate"):
        pair_values(log_probs, ref, mask, flipped, 0.1, 0.0)
    masked = mask.clone()
    masked[0, :] = 0
    with pytest.raises(ValueError, match="eligible"):
        pair_values(log_probs, ref, masked, roles, 0.1, 0.0)
