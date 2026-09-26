import torch
from omegaconf import OmegaConf
from skyrl_train.utils.policy_losses import LossScaling, compute_policy_objective, ppo_policy_loss

OLD = torch.full((4, 5), -1.0, dtype=torch.float64)
CURRENT = torch.tensor(
    [
        [-1.12, -0.78, -1.04, -0.65, -1.18],
        [-0.86, -1.24, -1.08, -1.00, -1.00],
        [-1.05, -0.91, -1.27, -0.96, -1.00],
        [-0.72, -1.13, -1.00, -1.00, -1.00],
    ],
    dtype=torch.float64,
)
BASE = torch.full((4, 5), -0.94, dtype=torch.float64)
ADVANTAGES = torch.tensor(
    [
        [0.7, -0.4, 0.9, -0.6, 0.3],
        [-0.8, 0.5, -0.3, 0.0, 0.0],
        [0.4, -0.9, 0.6, -0.2, 0.0],
        [-0.7, 0.8, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float64,
)
MASK = torch.tensor(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0],
    ],
    dtype=torch.float64,
)


def _config():
    return OmegaConf.create(
        {
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 5,
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "think_token_weight": 1.0,
            "use_entropy_loss": False,
            "entropy_loss_coef": 0.01,
            "use_kl_loss": True,
            "kl_loss_coef": 0.001,
            "kl_estimator_type": "k3",
            "use_tis": False,
            "tis_imp_ratio_cap": -1.0,
        }
    )


def _reference(actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Microbatch one gives every sequence equal weight, even when their valid
    # response lengths differ. This is the configured token_mean behavior for
    # Iceball's actual gradient-accumulation geometry.
    ratio = (actions - OLD).exp()
    clipped = ratio.clamp(0.8, 1.2)
    token_loss = -torch.minimum(ratio * ADVANTAGES, clipped * ADVANTAGES)
    kl_delta = (BASE - actions).clamp(-20, 20)
    kl = (kl_delta.exp() - kl_delta - 1).clamp(-10, 10)
    per_row = ((token_loss + 0.001 * kl) * mask).sum(dim=1) / mask.sum(dim=1)
    return per_row.mean()


def _backend_loss(actions: torch.Tensor, mask: torch.Tensor, scaling: LossScaling) -> torch.Tensor:
    outputs = []
    for row in range(actions.shape[0]):
        objective = compute_policy_objective(
            action_log_probs=actions[row : row + 1],
            old_action_log_probs=OLD[row : row + 1],
            base_action_log_probs=BASE[row : row + 1],
            advantages=ADVANTAGES[row : row + 1],
            loss_mask=mask[row : row + 1],
            rollout_logprobs=None,
            response_span_tags=None,
            token_entropy=torch.zeros_like(actions[row : row + 1]),
            config=_config(),
            policy_loss_fn=ppo_policy_loss,
            accumulation_steps=actions.shape[0],
            scaling=scaling,
        )
        outputs.append(objective.optimization_loss)
    total = torch.stack(outputs).sum()
    return total / actions.shape[0] if scaling is LossScaling.MEGATRON_PIPELINE else total


def test_iceball_masked_signed_grpo_matches_independent_microbatch_reference():
    expected_actions = CURRENT.clone().requires_grad_()
    expected = _reference(expected_actions, MASK)
    expected_grad = torch.autograd.grad(expected, expected_actions)[0]

    for scaling in (LossScaling.CALLER, LossScaling.MEGATRON_PIPELINE):
        actions = CURRENT.clone().requires_grad_()
        actual = _backend_loss(actions, MASK, scaling)
        actual_grad = torch.autograd.grad(actual, actions)[0]
        # The production ratio exp is evaluated in fp32 before converting to
        # the input dtype; the independent expression above stays in fp64.
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-8)
        assert torch.count_nonzero(actual_grad[MASK == 0]) == 0
        assert torch.count_nonzero(actual_grad[MASK != 0]) > 0

    corrupted_mask = MASK.clone()
    corrupted_mask[0, 4] = 0
    assert (_reference(CURRENT, corrupted_mask) - expected).abs() > 1e-3
