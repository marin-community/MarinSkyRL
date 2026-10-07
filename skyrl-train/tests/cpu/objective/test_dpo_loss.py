import math

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from skyrl_train.objective.dpo import DPOInputs, dpo_pair_values
from skyrl_train.objective.losses import PolicyLossInputs, dpo_policy_loss
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry


def make_inputs(seed=0, rows=4, cols=3, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    log_probs = torch.randn(rows, cols, generator=generator, dtype=dtype, requires_grad=True)
    ref = torch.randn(rows, cols, generator=generator, dtype=dtype)
    mask = torch.ones(rows, cols, dtype=dtype)
    roles = torch.tensor([1.0, -1.0], dtype=dtype).repeat(rows // 2)
    return log_probs, ref, mask, roles


def reference_loss(log_probs, ref, mask, roles, beta, smoothing):
    """TRL's sigmoid/robust pair formula computed pair-wise."""
    pairs = log_probs.reshape(-1, 2, log_probs.shape[-1])
    ref_pairs = ref.reshape(-1, 2, ref.shape[-1])
    deltas = ((pairs - ref_pairs) * mask.reshape(-1, 2, mask.shape[-1])).sum(-1)
    delta = deltas[:, 0] - deltas[:, 1]
    scale = 1.0 / (1.0 - 2.0 * smoothing)
    loss = scale * (
        -(1 - smoothing) * F.logsigmoid(beta * delta) - smoothing * F.logsigmoid(-beta * delta)
    )
    return loss.mean(), delta


@pytest.mark.parametrize("beta", [0.05, 0.1, 1.0])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_dpo_gradient_matches_reference_formula(beta, smoothing):
    log_probs, ref, mask, roles = make_inputs()
    values, _metrics = dpo_pair_values(
        PolicyLossInputs(log_probs, log_probs.detach(), None, torch.zeros_like(mask), mask, ref_log_probs=ref), DPOInputs(roles),
        beta, smoothing,
    )
    pairs = log_probs.shape[0] // 2
    surrogate = values.sum() / pairs
    actual = torch.autograd.grad(surrogate, log_probs)[0]
    expected_loss, _ = reference_loss(log_probs, ref, mask, roles, beta, smoothing)
    expected = torch.autograd.grad(expected_loss, log_probs)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("smoothing", [0.0, 0.2])
def test_dpo_metrics_match_closed_forms(smoothing):
    beta = 0.1
    log_probs, ref, mask, roles = make_inputs(seed=3)
    _values, metrics = dpo_pair_values(
        PolicyLossInputs(log_probs, log_probs.detach(), None, torch.zeros_like(mask), mask, ref_log_probs=ref), DPOInputs(roles),
        beta, smoothing,
    )
    expected_loss, delta = reference_loss(log_probs, ref, mask, roles, beta, smoothing)
    chosen_reward = beta * (
        (log_probs[0::2] - ref[0::2]).sum(-1)
    )
    rejected_reward = beta * ((log_probs[1::2] - ref[1::2]).sum(-1))
    assert metrics["dpo/loss"] == pytest.approx(float(expected_loss))
    assert metrics["dpo/accuracy"] == pytest.approx(float((delta > 0).float().mean()))
    assert metrics["dpo/margin"] == pytest.approx(float((chosen_reward - rejected_reward).mean()))
    assert metrics["dpo/chosen_reward"] == pytest.approx(float(chosen_reward.mean()))
    assert metrics["dpo/rejected_reward"] == pytest.approx(float(rejected_reward.mean()))
    if smoothing == 0.0:
        # At the reference policy the loss is log 2 and the surrogate row is zero.
        equal = PolicyLossInputs(ref, ref, None, torch.zeros_like(mask), mask, ref_log_probs=ref)
        zero_values, zero_metrics = dpo_pair_values(equal, DPOInputs(roles), beta, 0.0)
        assert zero_metrics["dpo/loss"] == pytest.approx(math.log(2.0))
        assert zero_values.sum().item() == pytest.approx(0.0, abs=1e-9)


def test_registered_loss_reads_config():
    log_probs, ref, mask, roles = make_inputs()
    config = OmegaConf.create({"dpo": {"beta": 0.3, "label_smoothing": 0.0}})
    inputs = PolicyLossInputs(log_probs, log_probs.detach(), None, torch.zeros_like(mask), mask, ref_log_probs=ref, dpo=DPOInputs(roles))
    direct_values, _ = dpo_pair_values(inputs, DPOInputs(roles), 0.3, 0.0)
    registered = dpo_policy_loss(inputs, config)
    torch.testing.assert_close(registered.values, direct_values)
    assert PolicyLossRegistry.get("dpo").spec.anchor.value == "none"


def test_dpo_rejects_broken_pairs():
    log_probs, ref, mask, roles = make_inputs()
    inputs = PolicyLossInputs(log_probs, log_probs.detach(), None, torch.zeros_like(mask), mask, ref_log_probs=ref)

    with pytest.raises(ValueError, match="reference"):
        dpo_pair_values(
            PolicyLossInputs(log_probs, log_probs.detach(), None, torch.zeros_like(mask), mask, dpo=DPOInputs(roles)),
            DPOInputs(roles),
            0.1,
            0.0,
        )

    with pytest.raises(ValueError, match="even microbatch"):
        dpo_pair_values(
            PolicyLossInputs(log_probs[:3], log_probs.detach()[:3], None, torch.zeros_like(mask[:3]), mask[:3], ref_log_probs=ref[:3]),
            DPOInputs(roles[:3]),
            0.1,
            0.0,
        )

    flipped = DPOInputs(torch.tensor([-1.0, 1.0, 1.0, -1.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="alternate"):
        dpo_pair_values(inputs, flipped, 0.1, 0.0)

    masked = mask.clone()
    masked[0, :] = 0
    with pytest.raises(ValueError, match="eligible"):
        dpo_pair_values(
            PolicyLossInputs(log_probs, log_probs.detach(), None, torch.zeros_like(mask), masked, ref_log_probs=ref),
            DPOInputs(roles),
            0.1,
            0.0,
        )
