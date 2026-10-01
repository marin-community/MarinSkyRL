"""Policy loss values, clip metrics, and loss reductions."""

import math

import pytest
import torch
from omegaconf import DictConfig

from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from skyrl_train.config.objective_spec import LossReduction
from skyrl_train.objective.losses import PolicyLossInputs
from skyrl_train.objective.reduction import reduce_to_step, step_counts


def _policy_loss(name):
    loss = PolicyLossRegistry.get(name)

    def evaluate(log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None):
        mask = torch.ones_like(log_probs) if loss_mask is None else loss_mask
        inputs = PolicyLossInputs(log_probs, old_log_probs, rollout_logprobs, advantages, mask)
        result = loss(inputs, config)
        counts = step_counts([mask], [mask], [], [advantages], config.max_seq_len, lambda value: value)
        reduced = reduce_to_step(
            result.values,
            mask,
            counts.policy,
            LossReduction(config.loss_reduction),
            max_seq_len=counts.max_seq_len,
            nonzero_advantage_rows=counts.nonzero_advantage_rows,
        )
        return reduced, result.metrics

    return evaluate


def _clipping_config(loss_name: str, *, eps_clip_low: float, eps_clip_high: float) -> DictConfig:
    return DictConfig(
        {
            "eps_clip_low": eps_clip_low,
            "eps_clip_high": eps_clip_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": loss_name,
            "loss_reduction": "sequence_mean",
            "max_seq_len": 2,
            "use_tis": False,
            "cispo": {
                "cispo_eps_clip_low": eps_clip_low,
                "cispo_eps_clip_high": eps_clip_high,
            },
        }
    )


@pytest.mark.parametrize("loss_name", ["regular", "gspo", "cispo"])
def test_clip_bounds_control_only_their_ratio_side(loss_name: str):
    loss_fn = _policy_loss(loss_name)
    old_log_probs = torch.zeros((2, 1))
    log_probs = torch.log(torch.tensor([[0.75], [1.10]]))
    advantages = torch.tensor([[-1.0], [1.0]])
    low_token = torch.tensor([[1.0], [0.0]])
    high_token = torch.tensor([[0.0], [1.0]])

    base = _clipping_config(loss_name, eps_clip_low=0.2, eps_clip_high=0.05)
    wider_low = _clipping_config(loss_name, eps_clip_low=0.3, eps_clip_high=0.05)
    wider_high = _clipping_config(loss_name, eps_clip_low=0.2, eps_clip_high=0.2)

    base_low_loss, _ = loss_fn(log_probs, old_log_probs, advantages, base, low_token)
    base_high_loss, _ = loss_fn(log_probs, old_log_probs, advantages, base, high_token)
    wider_low_low_loss, _ = loss_fn(log_probs, old_log_probs, advantages, wider_low, low_token)
    wider_low_high_loss, _ = loss_fn(log_probs, old_log_probs, advantages, wider_low, high_token)
    wider_high_low_loss, _ = loss_fn(log_probs, old_log_probs, advantages, wider_high, low_token)
    wider_high_high_loss, _ = loss_fn(log_probs, old_log_probs, advantages, wider_high, high_token)

    assert wider_low_low_loss.item() != pytest.approx(base_low_loss.item())
    assert wider_low_high_loss.item() == pytest.approx(base_high_loss.item())
    assert wider_high_low_loss.item() == pytest.approx(base_low_loss.item())
    assert wider_high_high_loss.item() != pytest.approx(base_high_loss.item())


@pytest.mark.parametrize("loss_name", ["regular", "gspo", "cispo"])
def test_policy_loss_reports_clip_decisions_and_pressure_by_ratio_side(loss_name: str):
    loss_fn = _policy_loss(loss_name)
    old_log_probs = torch.zeros((4, 1))
    log_probs = torch.log(torch.tensor([[0.75], [0.85], [1.10], [1.02]]))
    advantages = torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])

    _, metrics = loss_fn(
        log_probs,
        old_log_probs,
        advantages,
        _clipping_config(loss_name, eps_clip_low=0.2, eps_clip_high=0.05),
    )

    assert metrics == {
        "ppo_clip_ratio": pytest.approx(0.5),
        "ppo_clip_ratio_low": pytest.approx(0.25),
        "ppo_clip_ratio_high": pytest.approx(0.25),
        "ppo_clip_pressure_low": pytest.approx(0.25),
        "ppo_clip_pressure_high": pytest.approx(0.25),
        "ppo_ratio_exact_unit_fraction": pytest.approx(0.0),
    }


def test_policy_loss_dual_clip():
    # ratios ~= [0.5, 1.0, 10.0]; advantages positive, slightly negative, strongly negative.
    advantages = torch.tensor([[1.0, -1.0, -4.0]])
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]])
    log_probs = torch.tensor([[-1.69315, -1.0, -0.69741]])
    config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "dual_clip",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    loss, _ = _policy_loss("dual_clip")(log_probs, old_log_probs, advantages, config)

    # Per-token PPO losses max(-r*A, -clip(r)*A) = [-0.5, 1.0, 40.0]; the dual clip caps the
    # negative-advantage tokens at -A * clip_ratio_c: [-0.5, 1.0, 12.0] -> mean 12.5 / 3.
    assert loss.item() == pytest.approx(4.1667, abs=1e-4)


def test_behavior_clip_matches_regular_loss_on_policy():
    advantages = torch.tensor([[1.0, -1.0, 2.0]])
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]])
    log_probs = torch.tensor([[-1.2, -0.9, -2.7]])
    config = _clipping_config("regular", eps_clip_low=0.2, eps_clip_high=0.2)

    regular_loss, _ = _policy_loss("regular")(
        log_probs,
        old_log_probs,
        advantages,
        config,
        rollout_logprobs=old_log_probs,
    )
    config.policy_loss_type = "behavior_clip"
    behavior_loss, _ = _policy_loss("behavior_clip")(
        log_probs,
        old_log_probs,
        advantages,
        config,
        rollout_logprobs=old_log_probs,
    )

    torch.testing.assert_close(behavior_loss, regular_loss)


def test_behavior_clip_stops_resuppressing_stale_negative_advantage_token():
    log_probs = torch.tensor([[math.log(0.5)]], requires_grad=True)
    rollout_logprobs = torch.zeros_like(log_probs)
    old_log_probs = log_probs.detach().clone()
    advantages = torch.tensor([[-1.0]])
    config = _clipping_config("behavior_clip", eps_clip_low=0.2, eps_clip_high=0.2)

    loss, metrics = _policy_loss("behavior_clip")(
        log_probs,
        old_log_probs,
        advantages,
        config,
        rollout_logprobs=rollout_logprobs,
    )
    loss.backward()

    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs))
    assert metrics["ppo_clip_ratio_low"] == pytest.approx(1.0)


def test_policy_loss_cispo():
    # ratios ~= [0.5, 1.0, 10.0] clamp to [0.8, 1.0, 1.2].
    advantages = torch.tensor([[1.0, -1.0, -4.0]])
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]])
    log_probs = torch.tensor([[-1.69315, -1.0, -0.69741]])
    config = DictConfig(
        {
            "cispo": {"cispo_eps_clip_low": 0.2, "cispo_eps_clip_high": 0.2},
            "policy_loss_type": "cispo",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    loss, _ = _policy_loss("cispo")(log_probs, old_log_probs, advantages, config)

    # -A * clip(r) * logp = [1.35452, -1.0, -3.347568] -> mean -0.99768.
    assert loss.item() == pytest.approx(-0.99768266666, abs=1e-4)


def test_gspo_uses_masked_sequence_level_ratio():
    # Sequence 0: token log ratios [0.1, 0.3] -> sequence ratio e^0.2 = 1.2214, clipped to 1.2 for A=+1.
    # Sequence 1: token log ratios [-0.1, 0.1] -> ratio 1.0; the masked log ratio of 5.0 must not count.
    old_log_probs = torch.zeros(2, 3)
    log_probs = torch.tensor([[0.1, 0.3, 0.0], [-0.1, 0.1, 5.0]])
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    config = _clipping_config("gspo", eps_clip_low=0.2, eps_clip_high=0.2)

    loss, _ = _policy_loss("gspo")(log_probs, old_log_probs, advantages, config, loss_mask)

    # sequence_mean of per-token losses: (-1.2 + 2.0) / 2
    assert loss.item() == pytest.approx(0.4, abs=1e-6)


def test_clip_cov_zeroes_covariance_selected_token():
    # ratios [e^0.5, 1, 1]. Token 0 is PPO-clipped and so excluded from covariance selection; token 2 has
    # covariance (A - 0) * (logp - mean(logp)) = 1/6, the only value inside (lb, ub) = (0.1, 5).
    advantages = torch.tensor([[1.0, 0.0, -1.0]])
    old_log_probs = torch.full((1, 3), -1.0)
    log_probs = torch.tensor([[-0.5, -1.0, -1.0]])
    loss_mask = torch.ones(1, 3)
    config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "policy_loss_type": "clip_cov",
            "loss_reduction": "token_mean",
            "max_seq_len": 3,
            "clip_cov": {"clip_ratio": 0.5, "clip_cov_lb": 0.1, "clip_cov_ub": 5.0},
        }
    )

    loss, metrics = _policy_loss("clip_cov")(log_probs, old_log_probs, advantages, config, loss_mask)

    # Per-token PPO losses [-1.2, 0, 1]; token 2 is zeroed -> (-1.2 + 0 + 0) / 3.
    assert loss.item() == pytest.approx(-0.4, abs=1e-6)
    assert metrics["ppo_clip_ratio"] == pytest.approx(1 / 3)


def test_kl_cov_adds_kl_penalty_to_highest_covariance_token():
    # Covariance (A - 0) * (logp - mean(logp)) = [1/3, 0, 1/6]; kl_cov_frac selects the top token only.
    advantages = torch.tensor([[1.0, 0.0, -1.0]])
    old_log_probs = torch.full((1, 3), -1.0)
    log_probs = torch.tensor([[-0.5, -1.0, -1.0]])
    loss_mask = torch.ones(1, 3)
    config = DictConfig(
        {
            "policy_loss_type": "kl_cov",
            "loss_reduction": "token_mean",
            "max_seq_len": 3,
            "kl_cov": {"kl_cov_frac": 0.34, "ppo_kl_coef": 1.0},
        }
    )

    loss, _ = _policy_loss("kl_cov")(log_probs, old_log_probs, advantages, config, loss_mask)

    # -A * r = [-e^0.5, 0, 1]; token 0 adds |log r| = 0.5 -> (-e^0.5 + 0.5 + 1) / 3.
    assert loss.item() == pytest.approx(-0.0495738, abs=1e-6)


def test_sapo_policy_loss():
    # ratios [e^-0.5, e^0.2, e^-0.1]; tau_pos=1 for A>0, tau_neg=2 for A<=0.
    advantages = torch.tensor([[1.0, -1.0, 0.5]])
    old_log_probs = torch.full((1, 3), -1.0)
    log_probs = torch.tensor([[-1.5, -0.8, -1.1]])
    config = DictConfig(
        {
            "policy_loss_type": "sapo",
            "loss_reduction": "sequence_mean",
            "max_seq_len": 4,
            "sapo": {"tau_pos": 1.0, "tau_neg": 2.0},
        }
    )

    loss, _ = _policy_loss("sapo")(log_probs, old_log_probs, advantages, config)

    # gate(r, tau) = sigmoid(tau * (r - 1)) * 4 / tau; per-token -gate * A = [-1.61153, 1.21785, -0.95245].
    assert loss.item() == pytest.approx(-0.4487099, abs=1e-6)


def test_tis_graceful_degrade_on_none_logprobs():
    """Fix A: use_tis=True but a batch with no rollout logprobs must degrade to
    the standard (non-TIS) policy loss for THAT batch instead of crashing.

    Guards:
      1. use_tis=True + rollout_logprobs=None  == use_tis=False loss (TIS skipped).
      2. use_tis=True + rollout_logprobs given  != the degraded loss (TIS applied),
         i.e. the importance ratio is NOT silently dropped when logprobs ARE present.
    """
    device = "cpu"

    advantages = torch.tensor([[1.0, -1.0, 2.0]], device=device)
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]], device=device)
    log_probs = torch.tensor([[-1.2, -0.9, -2.7]], device=device)
    # rollout logprobs deliberately offset from old_log_probs so the TIS ratio != 1.
    rollout_logprobs = torch.tensor([[-0.5, -1.5, -2.0]], device=device)

    base_cfg = {
        "eps_clip_low": 0.2,
        "eps_clip_high": 0.2,
        "clip_ratio_c": 3.0,
        "policy_loss_type": "regular",
        "loss_reduction": "token_mean",
        "max_seq_len": 4,
        "tis_imp_ratio_cap": 2.0,
    }
    loss_fn = _policy_loss("regular")

    # Reference: TIS off.
    cfg_off = DictConfig({**base_cfg, "use_tis": False})
    loss_off, _ = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=cfg_off,
        rollout_logprobs=None,
    )

    # 1. TIS on but logprobs missing -> must equal the TIS-off loss (degraded).
    cfg_on = DictConfig({**base_cfg, "use_tis": True})
    loss_degraded, _ = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=cfg_on,
        rollout_logprobs=None,
    )
    torch.testing.assert_close(loss_degraded, loss_off, rtol=1e-6, atol=1e-8)

    # 2. TIS on WITH logprobs -> ratio applied -> loss differs from degraded.
    loss_tis, _ = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=cfg_on,
        rollout_logprobs=rollout_logprobs,
    )
    assert not torch.allclose(loss_tis, loss_off), "TIS ratio should be applied when rollout_logprobs is present"
