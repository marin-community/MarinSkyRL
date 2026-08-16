"""
Run with:
uv run --isolated --group dev --extra cpu pytest tests/cpu/utils/test_policy_optimization.py
"""

import torch
import math
import pytest
from omegaconf import OmegaConf
from skyrl_train.utils.loss_reduction import compute_global_loss_denom, count_nonzero_advantage_seqs, reduce_loss
from skyrl_train.utils.policy_math import compute_approx_kl
from skyrl_train.utils.policy_losses import LossScaling, compute_policy_objective
from skyrl_train.utils.advantage_estimators import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_advantages_and_returns,
    compute_reinforce_plus_plus_outcome_advantage,
    compute_rloo_outcome_advantage,
)
from skyrl_train.utils.kl_controllers import AdaptiveKLController, FixedKLController
from skyrl_train.utils.algorithm_registry import (
    AdvantageEstimatorRegistry,
    NoGroupAdvantage,
    register_advantage_estimator,
    PolicyLossRegistry,
    register_policy_loss,
)
from skyrl_train.utils.importance_ratio_diagnostics import compute_tis_diagnostics, TIS_DIAG_KEYS
from skyrl_train.utils.utils import validate_cfg
import numpy as np


@pytest.fixture
def dummy_data():
    log_probs = torch.tensor([[0.2, 0.3, 0.5]])
    log_probs_base = torch.tensor([[0.1, 0.2, 0.4]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])  # last value masked out
    return log_probs, log_probs_base, mask


@pytest.fixture
def advantage_test_data():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 1.0, 1.5]])
    response_mask = torch.tensor([[1.0, 1.0, 1.0]])
    index = np.array(["0", "0", "0"])
    return rewards, values, response_mask, index


def test_compute_approx_kl(dummy_data):
    log_probs, log_probs_base, mask = dummy_data
    kl = compute_approx_kl(log_probs, log_probs_base, mask, kl_estimator_type="k1")

    expected_kl = (log_probs - log_probs_base) * mask
    assert torch.allclose(kl, expected_kl), "KL approximation should be log-prob diff masked"

    kl_abs = compute_approx_kl(log_probs, log_probs_base, mask, kl_estimator_type="abs")
    expected_abs = (log_probs - log_probs_base).abs() * mask
    assert torch.allclose(kl_abs, expected_abs), "KL approximation should be abs(log-prob diff) masked"

    kl_k2 = compute_approx_kl(log_probs, log_probs_base, mask, kl_estimator_type="k2")
    expected_k2 = 0.5 * (log_probs - log_probs_base).square() * mask
    assert torch.allclose(kl_k2, expected_k2, atol=1e-4), "k2 estimator is not correct"

    kl_k3 = compute_approx_kl(log_probs, log_probs_base, mask, kl_estimator_type="k3")
    log_ratio = log_probs - log_probs_base
    expected_k3 = (torch.exp(-log_ratio) - 1 + log_ratio) * mask
    assert torch.allclose(kl_k3, expected_k3, atol=1e-4), "k3 estimator is not correct"


def test_compute_reinforce_plus_plus_outcome_advantage_returns_and_masking():
    """REINFORCE++ returns should be discounted sums with reset after EOS; advantages masked."""
    token_level_rewards = torch.tensor([[1.0, 2.0, 3.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0]])  # EOS after second token

    adv, ret = compute_reinforce_plus_plus_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        gamma=1.0,
    )

    expected_ret = torch.tensor([[3.0, 2.0, 3.0]])

    assert ret.shape == token_level_rewards.shape
    assert torch.allclose(ret, expected_ret, atol=1e-5)
    # advantages are whitened and then masked; masked positions should be zero
    assert adv.shape == token_level_rewards.shape
    assert torch.allclose(adv * (1 - response_mask), torch.zeros_like(adv))


def test_compute_reinforce_plus_plus_outcome_advantage_gamma():
    """REINFORCE++ returns should reflect gamma discounting."""
    token_level_rewards = torch.tensor([[1.0, 2.0, 3.0]])
    response_mask = torch.ones_like(token_level_rewards)

    adv, ret = compute_reinforce_plus_plus_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        gamma=0.5,
    )

    expected_ret = torch.tensor([[2.75, 3.50, 3.00]])

    assert ret.shape == token_level_rewards.shape
    assert torch.allclose(ret, expected_ret, atol=1e-5)
    assert adv.shape == token_level_rewards.shape


def test_compute_rloo_outcome_advantage_basic():
    """RLOO should produce leave-one-out centered scores per group, broadcast across tokens."""
    # Three groups: [6.0, 3.0] -> [3.0, -3.0], [9.0, 12.0] -> [-3.0, 3.0]
    # [1.0] -> [0.0] (since there's only one response, the advantage is 0)
    token_level_rewards = torch.tensor(
        [
            [0.0, 0.0, 6.0],  # sum = 6.0, group 0
            [0.0, 0.0, 3.0],  # sum = 3.0, group 0
            [0.0, 0.0, 9.0],  # sum = 9.0, group 1
            [0.0, 0.0, 12.0],  # sum = 12.0, group 1
            [0.0, 0.0, 1.0],  # sum = 0.0, group 2
        ]
    )
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array([0, 0, 1, 1, 2])

    adv, ret = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    expected = torch.tensor([3.0, -3.0, -3.0, 3.0, 0.0]).unsqueeze(-1) * response_mask

    assert adv.shape == token_level_rewards.shape
    assert torch.allclose(adv, ret), "Advantages and returns should be equal with RLOO"
    assert torch.allclose(adv, expected, atol=1e-5)


def test_compute_grpo_outcome_advantage(advantage_test_data):
    rewards, _, response_mask, index = advantage_test_data

    adv, ret = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
    )

    assert adv.shape == rewards.shape
    assert ret.shape == rewards.shape
    assert torch.allclose(adv, ret), "Advantages and returns should be equal with GRPO"


def test_compute_grpo_outcome_advantage_norm_std_false():
    """Test GRPO advantage computation with grpo_norm_by_std=False."""
    # Two groups: [6.0, 3.0] mean=4.5, [9.0, 12.0] mean=10.5
    token_level_rewards = torch.tensor(
        [
            [1.0, 2.0, 3.0],  # sum = 6.0, group 0
            [1.0, 1.0, 1.0],  # sum = 3.0, group 0
            [3.0, 3.0, 3.0],  # sum = 9.0, group 1
            [4.0, 4.0, 4.0],  # sum = 12.0, group 1
        ]
    )
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array([0, 0, 1, 1])

    adv, ret = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        grpo_norm_by_std=False,
    )

    # Expected: [6.0-4.5, 3.0-4.5, 9.0-10.5, 12.0-10.5] = [1.5, -1.5, -1.5, 1.5]
    expected = torch.tensor([1.5, -1.5, -1.5, 1.5]).unsqueeze(-1) * response_mask

    assert adv.shape == token_level_rewards.shape
    assert torch.allclose(adv, ret), "Advantages and returns should be equal with GRPO"
    assert torch.allclose(adv, expected, atol=1e-5), f"Expected {expected}, got {adv}"


def test_compute_gae_advantage_return(advantage_test_data):
    rewards, values, response_mask, index = advantage_test_data

    adv, ret = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=1.0,
        lambd=1.0,  # no discounting for simplicity
    )

    expected_ret = torch.tensor([[6.0, 5.0, 3.0]])

    # The advantages will be whitened, so we just check the shape and that they're not all zeros
    assert adv.shape == rewards.shape
    assert not torch.allclose(adv, torch.zeros_like(adv))
    assert ret.shape == expected_ret.shape
    assert torch.allclose(ret, expected_ret, atol=1e-5)


def test_compute_gae_advantage_return_with_masking(advantage_test_data):
    rewards, values, _, _ = advantage_test_data
    response_mask = torch.tensor([[1.0, 0.0, 1.0]])  # Mask out the second token

    adv, ret = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=1.0,
        lambd=1.0,  # no discounting for simplicity
    )

    # The returns should be reversed cumulative rewards
    expected_ret = torch.tensor([[6.0, 5.0, 3.0]])
    expected_adv = torch.tensor([[0.7071, 0.1768, -0.7071]])

    assert torch.allclose(ret, expected_ret, atol=1e-5)
    assert torch.allclose(adv, expected_adv, atol=1e-4)


def test_compute_gae_advantage_return_gamma(advantage_test_data):
    rewards, values, response_mask, _ = advantage_test_data

    _, ret = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=0.5,
        lambd=1.0,
    )

    expected_ret = torch.tensor([[2.7500, 3.5000, 3.0000]])
    assert torch.allclose(ret, expected_ret, atol=1e-5)


def test_compute_gae_advantage_return_lam(advantage_test_data):
    rewards, values, response_mask, _ = advantage_test_data

    _, ret = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        lambd=0.5,
        gamma=1.0,
    )

    expected_ret = torch.tensor([[3.6250, 4.2500, 3.0000]])
    assert torch.allclose(ret, expected_ret, atol=1e-5)


def test_reduce_loss():
    """Test the reduce_loss function with different reduction types."""
    # Test data: 2x3 loss tensor with different valid token counts per sequence
    loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])  # seq0 has 3 tokens, seq1 has 1 token

    # Test token_mean: sum all valid losses / count valid tokens
    # Valid losses: [1.0, 2.0, 3.0, 4.0], mean = 10.0/4 = 2.5
    result_token = reduce_loss(loss, loss_mask, "token_mean")
    expected_token = torch.tensor(2.5)
    assert torch.allclose(result_token, expected_token), f"Expected {expected_token}, got {result_token}"

    # Test sequence_mean: mean of per-sequence means
    # Seq 0: (1.0 + 2.0 + 3.0) / 3 = 2.0, Seq 1: 4.0 / 1 = 4.0, batch mean = (2.0 + 4.0) / 2 = 3.0
    result_seq = reduce_loss(loss, loss_mask, "sequence_mean")
    expected_seq = torch.tensor(3.0)
    assert torch.allclose(result_seq, expected_seq), f"Expected {expected_seq}, got {result_seq}"

    # Test seq_mean_token_sum_norm: sum per sequence / max_len, then batch mean
    # Seq 0: (1.0 + 2.0 + 3.0) / 4 = 1.5, Seq 1: 4.0 / 4 = 1.0, batch mean = (1.5 + 1.0) / 2 = 1.25
    max_seq_len = 4
    result_max = reduce_loss(loss, loss_mask, "seq_mean_token_sum_norm", max_seq_len)
    expected_max = torch.tensor(1.25)
    assert torch.allclose(result_max, expected_max), f"Expected {expected_max}, got {result_max}"


def _validatable_dummy_config():
    """A dummy config that passes validate_batch_sizes so validate_cfg reaches the
    loss_reduction allow-list (single-GPU placement, all batch sizes == 1)."""
    from omegaconf import OmegaConf
    from tests.cpu.util import example_dummy_config

    cfg = example_dummy_config()
    OmegaConf.update(
        cfg,
        "trainer",
        {
            "train_batch_size": 1,
            "policy_mini_batch_size": 1,
            "critic_mini_batch_size": 1,
            "micro_train_batch_size_per_gpu": 1,
            "micro_forward_batch_size_per_gpu": 1,
            "placement": {
                "policy_num_nodes": 1,
                "policy_num_gpus_per_node": 1,
                "critic_num_nodes": 1,
                "critic_num_gpus_per_node": 1,
                "ref_num_nodes": 1,
                "ref_num_gpus_per_node": 1,
            },
        },
    )
    return cfg


@pytest.mark.parametrize(
    "loss_reduction",
    ["token_mean", "sequence_mean", "seq_mean_token_sum_norm", "seq_mean_token_sum_norm_global"],
)
def test_validate_cfg_accepts_all_loss_reductions(loss_reduction):
    """Config-validation smoke: validate_cfg must NOT reject any supported loss_reduction.

    Regression guard for the arm1 failure where `seq_mean_token_sum_norm_global`
    was registered in reduce_loss + compute_policy_loss but rejected by the
    hardcoded allow-list in validate_cfg (utils.py). The minimal dummy config may
    still trip later (unrelated) placement/colocation asserts, so we only require
    that the *loss_reduction allow-list* never fires for a supported value.
    """
    pytest.importorskip("hydra")

    cfg = _validatable_dummy_config()
    OmegaConf.update(cfg, "trainer.algorithm.loss_reduction", loss_reduction)
    try:
        validate_cfg(cfg)
    except (AssertionError, ValueError) as e:
        assert "invalid loss_reduction" not in str(e), (
            f"supported loss_reduction {loss_reduction!r} was rejected by the allow-list: {e}"
        )


def test_global_loss_denom_driver_matches_allreduce_sum():
    """The DRIVER-side collective-free global loss denominator Z must be BIT-IDENTICAL
    to the historical in-worker ``all_reduce(sum)`` of per-rank ``local_num_seqs`` over
    the full policy PG. This is the objective-preservation proof for the 80B gs1 NCCL
    wedge fix (worker.py seq_mean_token_sum_norm_global normalizer, collective #288606):
    the fix moves the collective off the async ppo_train hot path to a driver precompute
    and MUST NOT change Z.
    """
    torch.manual_seed(0)
    max_seq_len = 4096
    # A range of (world_size, dp_size) mesh geometries, including the 80B
    # EP8xFSDP8xCP1 = 64-rank shape that hit the wedge.
    for world_size, dp_size in [(64, 2), (64, 8), (64, 16), (8, 4), (16, 16), (4, 1)]:
        ranks_per_dp_group = world_size // dp_size
        # Synthetic full-batch advantages: rows divisible by dp_size, with a deterministic
        # subset of all-zero rows (excluded / zero-variance) so the nonzero-seq count is
        # non-trivial and < n_rows.
        n_rows = dp_size * 5
        resp_len = 7
        adv = torch.randn(n_rows, resp_len)
        adv[::3] = 0.0

        # Reference: emulate the historical per-rank all_reduce(sum). MeshDispatch splits
        # the full batch into dp_size disjoint row-chunks; every rank in a dp-group holds
        # the SAME chunk, so summing local counts over ALL world_size ranks ==
        # ranks_per_dp_group * (per-chunk counts summed over the dp groups).
        chunks = torch.chunk(adv, dp_size, dim=0)
        assert len(chunks) == dp_size
        summed_over_ranks = 0.0
        for dp in range(dp_size):
            summed_over_ranks += ranks_per_dp_group * count_nonzero_advantage_seqs(chunks[dp])
        ref_Z = max(summed_over_ranks, 1.0) * max_seq_len

        # New: driver-side collective-free.
        new_Z = compute_global_loss_denom(adv, max_seq_len, ranks_per_dp_group)

        assert new_Z == ref_Z, f"(world={world_size}, dp={dp_size}) Z mismatch: {new_Z} != {ref_Z}"

    # All-zero-advantage batch -> clamp(min=1) path still yields a valid denom (matches
    # the legacy max(global_num_seqs, 1.0) clamp).
    adv_zero = torch.zeros(8, 5)
    assert compute_global_loss_denom(adv_zero, max_seq_len, 4) == 1.0 * max_seq_len


def _mask_sum_policy_loss(
    log_probs,
    old_log_probs,
    advantages,
    config,
    loss_mask=None,
    rollout_logprobs=None,
    global_loss_denom=None,
):
    del old_log_probs, advantages, config, rollout_logprobs, global_loss_denom
    return (log_probs * loss_mask).sum(), {}


@pytest.mark.parametrize("loss_reduction", ["token_mean", "seq_mean_token_sum_norm_global"])
def test_policy_objective_scheduler_and_caller_scaling_have_gradient_parity(loss_reduction):
    config = OmegaConf.create(
        {
            "loss_reduction": loss_reduction,
            "think_token_weight": 1.0,
            "use_entropy_loss": False,
            "entropy_loss_coef": 0.0,
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "kl_estimator_type": "k1",
            "use_tis": False,
            "tis_imp_ratio_cap": 2.0,
        }
    )
    accumulation_steps = 3
    common = {
        "old_action_log_probs": torch.zeros(1, 2),
        "base_action_log_probs": None,
        "advantages": torch.ones(1, 2),
        "loss_mask": torch.ones(1, 2),
        "rollout_logprobs": None,
        "response_span_tags": None,
        "token_entropy": torch.zeros(1, 2),
        "config": config,
        "policy_loss_fn": _mask_sum_policy_loss,
        "accumulation_steps": accumulation_steps,
        "global_loss_denom": 12.0,
    }

    caller_log_probs = torch.tensor([[1.0, 2.0]], requires_grad=True)
    caller = compute_policy_objective(
        action_log_probs=caller_log_probs,
        scaling=LossScaling.CALLER,
        **common,
    )
    caller.optimization_loss.backward()

    scheduler_log_probs = torch.tensor([[1.0, 2.0]], requires_grad=True)
    scheduler = compute_policy_objective(
        action_log_probs=scheduler_log_probs,
        scaling=LossScaling.MEGATRON_PIPELINE,
        **common,
    )
    (scheduler.optimization_loss / accumulation_steps).backward()

    torch.testing.assert_close(scheduler_log_probs.grad, caller_log_probs.grad, rtol=0, atol=0)
    assert caller.unscaled_loss.item() == pytest.approx(3.0)
    assert scheduler.unscaled_loss.item() == pytest.approx(3.0)
    expected_caller_loss = 3.0 if loss_reduction == "seq_mean_token_sum_norm_global" else 1.0
    expected_scheduler_loss = 9.0 if loss_reduction == "seq_mean_token_sum_norm_global" else 3.0
    assert caller.optimization_loss.item() == pytest.approx(expected_caller_loss)
    assert scheduler.optimization_loss.item() == pytest.approx(expected_scheduler_loss)
    assert "global_loss_denom" not in config


def test_policy_objective_applies_think_weight_before_policy_loss():
    config = OmegaConf.create(
        {
            "loss_reduction": "token_mean",
            "think_token_weight": 0.25,
            "use_entropy_loss": False,
            "entropy_loss_coef": 0.0,
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "kl_estimator_type": "k1",
            "use_tis": False,
            "tis_imp_ratio_cap": 2.0,
        }
    )
    result = compute_policy_objective(
        action_log_probs=torch.ones(1, 4),
        old_action_log_probs=torch.zeros(1, 4),
        base_action_log_probs=None,
        advantages=torch.ones(1, 4),
        loss_mask=torch.ones(1, 4),
        rollout_logprobs=None,
        response_span_tags=torch.tensor([[1, 0, 1, 0]]),
        token_entropy=torch.zeros(1, 4),
        config=config,
        policy_loss_fn=_mask_sum_policy_loss,
        accumulation_steps=1,
        scaling=LossScaling.CALLER,
    )

    assert result.policy_loss.item() == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("config_path", "invalid_value", "error"),
    [
        ("trainer.algorithm.loss_reduction", "definitely_not_a_reduction", "invalid loss_reduction"),
        ("trainer.policy.grug_query_bias_update_mode", "blend", "invalid grug_query_bias_update_mode"),
    ],
)
def test_validate_cfg_rejects_unknown_config_choice(config_path, invalid_value, error):
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf
    from skyrl_train.utils.utils import validate_cfg

    cfg = _validatable_dummy_config()
    OmegaConf.update(cfg, config_path, invalid_value)
    with pytest.raises(AssertionError, match=error):
        validate_cfg(cfg)


def test_validate_cfg_requires_grug_query_bias_update_mode():
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import validate_cfg

    cfg = _validatable_dummy_config()
    del cfg.trainer.policy.grug_query_bias_update_mode

    with pytest.raises(AssertionError, match="missing required policy configuration: grug_query_bias_update_mode"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_stacked_behavior_clip_and_tis():
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import validate_cfg

    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.policy_loss_type = "behavior_clip"
    cfg.trainer.algorithm.use_tis = True

    with pytest.raises(ValueError, match="cannot be combined with use_tis"):
        validate_cfg(cfg)


def test_validate_cfg_materializes_rloo_n_group_invariant():
    cfg = _validatable_dummy_config()
    cfg.generator.n_samples_per_prompt = 2
    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_pipeline_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.trainer.algorithm.advantage_estimator = "rloo_n"
    cfg.trainer.algorithm.group_advantage_min_size = 2

    validate_cfg(cfg)

    assert OmegaConf.to_container(cfg.trainer.algorithm.resolved_group_advantage) == {
        "kind": "minimum_baseline_eligible",
        "physical_group_size": 2,
        "minimum_group_size": 2,
    }


def test_adaptive_kl_controller_update():
    controller = AdaptiveKLController(init_kl_coef=0.2, target=0.1, horizon=100)
    controller.update(current=0.2, n_steps=10)

    # Expected error: (0.2 / 0.1 - 1) = 1 → clipped to 0.2
    # Mult = 1 + 0.2 * 10 / 100 = 1.02
    expected = 0.2 * 1.02
    assert math.isclose(controller.value, expected, rel_tol=1e-5)


def test_fixed_kl_controller():
    controller = FixedKLController(kl_coef=0.1)
    controller.update(current=1.0, n_steps=10)
    assert controller.value == 0.1  # Should remain unchanged


def test_base_function_registry_registration_and_retrieval():
    """Test basic registration and retrieval functionality of BaseFunctionRegistry."""

    def dummy_function(**kwargs):
        return torch.zeros_like(kwargs["token_level_rewards"]), torch.zeros_like(kwargs["token_level_rewards"])

    # Register function
    AdvantageEstimatorRegistry.register("test_basic", dummy_function, group_contract=NoGroupAdvantage())

    # Test retrieval
    retrieved_func = AdvantageEstimatorRegistry.get("test_basic")
    assert retrieved_func == dummy_function

    # Test it's in available list
    assert "test_basic" in AdvantageEstimatorRegistry.list_available()

    # Clean up
    AdvantageEstimatorRegistry.unregister("test_basic")


def test_advantage_estimator_registration_requires_group_contract():
    def dummy_function(**kwargs):
        return None, None

    with pytest.raises(ValueError, match="must declare a group_contract"):
        AdvantageEstimatorRegistry.register("missing_contract", dummy_function)


def test_base_function_registry_error_handling():
    """Test error handling in BaseFunctionRegistry."""

    def dummy_function(**kwargs):
        return None, None

    # Test getting non-existent function
    with pytest.raises(ValueError, match="Unknown advantage estimator"):
        AdvantageEstimatorRegistry.get("non_existent")

    # Test unregistering non-existent function
    with pytest.raises(ValueError, match="not registered"):
        AdvantageEstimatorRegistry.unregister("non_existent")

    # Test duplicate registration
    AdvantageEstimatorRegistry.register("test_dup", dummy_function, group_contract=NoGroupAdvantage())
    with pytest.raises(ValueError, match="already registered"):
        AdvantageEstimatorRegistry.register("test_dup", dummy_function, group_contract=NoGroupAdvantage())

    # Clean up
    AdvantageEstimatorRegistry.unregister("test_dup")


def test_base_registry_unregister():
    """Test unregistration functionality."""

    def dummy_function(**kwargs):
        return torch.zeros_like(kwargs["token_level_rewards"]), torch.zeros_like(kwargs["token_level_rewards"])

    # Register and verify
    AdvantageEstimatorRegistry.register("test_unregister", dummy_function, group_contract=NoGroupAdvantage())
    assert "test_unregister" in AdvantageEstimatorRegistry.list_available()

    # Unregister and verify
    AdvantageEstimatorRegistry.unregister("test_unregister")
    assert "test_unregister" not in AdvantageEstimatorRegistry.list_available()


def test_advantage_estimator_registry_specific():
    """Test AdvantageEstimatorRegistry-specific functionality."""

    @register_advantage_estimator("test_decorator", group_contract=NoGroupAdvantage())
    def decorated_estimator(**kwargs):
        return torch.ones_like(kwargs["token_level_rewards"]), torch.ones_like(kwargs["token_level_rewards"])

    # Test decorator worked
    assert "test_decorator" in AdvantageEstimatorRegistry.list_available()
    retrieved = AdvantageEstimatorRegistry.get("test_decorator")
    assert retrieved == decorated_estimator

    # Test integration with compute_advantages_and_returns
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    response_mask = torch.tensor([[1.0, 1.0, 1.0]])
    index = np.array(["0", "0", "0"])

    adv, ret = compute_advantages_and_returns(
        token_level_rewards=rewards, response_mask=response_mask, index=index, adv_estimator="test_decorator", config={}
    )

    assert torch.allclose(adv, torch.ones_like(rewards))
    assert torch.allclose(ret, torch.ones_like(rewards))

    # Clean up
    AdvantageEstimatorRegistry.unregister("test_decorator")


def test_policy_loss_registry_specific():
    """Test PolicyLossRegistry-specific functionality."""
    from omegaconf import DictConfig

    @register_policy_loss("test_policy_decorator")
    def decorated_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_log_probs=None):
        return torch.tensor(1.5), {"ppo_clip_ratio": 0.3}

    # Test decorator worked
    assert "test_policy_decorator" in PolicyLossRegistry.list_available()
    retrieved = PolicyLossRegistry.get("test_policy_decorator")
    assert retrieved == decorated_policy_loss

    # Test function execution
    config = DictConfig({"policy_loss_type": "test_policy_decorator"})
    loss, metrics = retrieved(
        log_probs=torch.tensor([[0.1]]),
        old_log_probs=torch.tensor([[0.2]]),
        advantages=torch.tensor([[1.0]]),
        config=config,
    )
    assert loss.item() == 1.5
    assert metrics["ppo_clip_ratio"] == 0.3

    # Test error message includes "Policy loss"
    with pytest.raises(ValueError, match="Unknown policy loss"):
        PolicyLossRegistry.get("non_existent_policy")

    # Clean up
    PolicyLossRegistry.unregister("test_policy_decorator")


def test_package_initialization_registers_complete_builtin_algorithm_sets():
    assert set(PolicyLossRegistry.list_available()) >= {
        "regular",
        "dual_clip",
        "gspo",
        "cispo",
        "clip_cov",
        "kl_cov",
        "sapo",
    }
    assert set(AdvantageEstimatorRegistry.list_available()) >= {
        "gae",
        "grpo",
        "rloo",
        "rloo_n",
        "rloo_n_pbs",
        "reinforce++",
    }


def test_validate_cfg_preserves_custom_policy_loss():
    def custom_policy_loss(*args, **kwargs):
        return torch.tensor(0.0), {}

    PolicyLossRegistry.register("custom_policy", custom_policy_loss)
    cfg = _validatable_dummy_config()
    OmegaConf.update(cfg, "trainer.algorithm.policy_loss_type", "custom_policy")
    OmegaConf.update(cfg, "generator.num_inference_engines", 1)
    OmegaConf.update(cfg, "generator.inference_engine_tensor_parallel_size", 1)
    OmegaConf.update(cfg, "generator.inference_engine_pipeline_parallel_size", 1)
    OmegaConf.update(cfg, "generator.inference_engine_data_parallel_size", 1)
    try:
        validate_cfg(cfg)
        assert PolicyLossRegistry.get("custom_policy") is custom_policy_loss
    finally:
        PolicyLossRegistry.unregister("custom_policy")


def _remove_registry_entries(registry, *names: str) -> None:
    for name in names:
        if name in registry.list_available():
            registry.unregister(name)
    registry.shutdown_actor()


@pytest.mark.usefixtures("ray_module")
def test_registry_cross_ray_process():
    """Test that registry works with Ray and that functions can be retrieved and called from different processes"""
    try:
        import ray
        from omegaconf import DictConfig

        # Create test functions
        def test_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None):
            return torch.tensor(2.0), {"ppo_clip_ratio": 0.5}

        def test_policy_loss_2(log_probs, old_log_probs, advantages, config, loss_mask=None):
            return torch.tensor(3.0), {"ppo_clip_ratio": 0.6}

        def test_advantage_estimator(**kwargs):
            rewards = kwargs["token_level_rewards"]
            return rewards * 2, rewards * 3

        # Test basic registration and retrieval
        PolicyLossRegistry.register("cross_process_test", test_policy_loss)
        AdvantageEstimatorRegistry.register(
            "cross_process_adv_test", test_advantage_estimator, group_contract=NoGroupAdvantage()
        )

        # Test Ray integration
        @ray.remote
        def test_ray_registry_access():
            policy_loss = PolicyLossRegistry.get("cross_process_test")
            adv_estimator = AdvantageEstimatorRegistry.get("cross_process_adv_test")

            loss, metrics = policy_loss(
                log_probs=torch.tensor([[0.1]]),
                old_log_probs=torch.tensor([[0.2]]),
                advantages=torch.tensor([[1.0]]),
                config=DictConfig({"policy_loss_type": "cross_process_test"}),
            )

            adv, ret = adv_estimator(
                token_level_rewards=torch.tensor([[1.0, 2.0]]),
                response_mask=torch.tensor([[1.0, 1.0]]),
                index=np.array(["0", "0"]),
            )
            return loss, metrics, adv, ret

        # Run Ray task
        loss, metrics, adv, ret = ray.get(test_ray_registry_access.remote())
        assert loss.item() == 2.0
        assert metrics["ppo_clip_ratio"] == 0.5
        assert adv.shape == torch.Size([1, 2])
        assert ret.shape == torch.Size([1, 2])

        # test that registration works after ray init as well
        PolicyLossRegistry.register("cross_process_test_2", test_policy_loss_2)
        loss_2, metrics_2 = PolicyLossRegistry.get("cross_process_test_2")(
            log_probs=torch.tensor([[0.1]]),
            old_log_probs=torch.tensor([[0.2]]),
            advantages=torch.tensor([[1.0]]),
            config=DictConfig({"policy_loss_type": "cross_process_test_2"}),
        )
        assert loss_2.item() == 3.0
        assert metrics_2["ppo_clip_ratio"] == 0.6
    finally:
        _remove_registry_entries(PolicyLossRegistry, "cross_process_test", "cross_process_test_2")
        _remove_registry_entries(AdvantageEstimatorRegistry, "cross_process_adv_test")


@pytest.mark.usefixtures("ray_module")
def test_registry_named_actor_creation():
    """Test that the registry creates named Ray actors and properly serializes functions."""
    try:
        import ray

        def test_func(**kwargs):
            rewards = kwargs["token_level_rewards"]
            return rewards * 2, rewards * 3

        # Register function (should create/use named actor)
        AdvantageEstimatorRegistry.register("named_actor_test", test_func, group_contract=NoGroupAdvantage())

        # Verify local retrieval works
        retrieved = AdvantageEstimatorRegistry.get("named_actor_test")
        assert retrieved == test_func

        # Verify named actor exists and contains function
        actor = ray.get_actor(AdvantageEstimatorRegistry._actor_name)
        assert actor is not None

        available_in_actor = ray.get(actor.list_available.remote())
        assert "named_actor_test" in available_in_actor

        # Verify function serialization/deserialization
        serialized_func = ray.get(actor.get.remote("named_actor_test"))
        assert serialized_func is not None

        import cloudpickle

        deserialized_func = cloudpickle.loads(serialized_func)

        # Test deserialized function works
        test_rewards = torch.tensor([[1.0, 2.0]])
        result = deserialized_func(
            token_level_rewards=test_rewards,
            response_mask=torch.tensor([[1.0, 1.0]]),
            index=np.array(["0", "0"]),
        )

        assert torch.allclose(result[0], test_rewards * 2)
        assert torch.allclose(result[1], test_rewards * 3)

    finally:
        _remove_registry_entries(AdvantageEstimatorRegistry, "named_actor_test")


@pytest.mark.usefixtures("ray_module")
def test_registry_reconnects_after_ray_shutdown():
    """
    Test that the registry reconnects properly after Ray is shut down.

    This mimics when we run multiple unit tests in a row with ray inits and shutdowns.
    """

    def _register_func_and_verify():
        """Register a function and verify it works."""

        def test_func(**kwargs):
            rewards = kwargs["token_level_rewards"]
            return rewards * 2, rewards * 3

        AdvantageEstimatorRegistry.register("named_actor_test", test_func, group_contract=NoGroupAdvantage())
        retrieved = AdvantageEstimatorRegistry.get("named_actor_test")
        assert retrieved == test_func
        actor = ray.get_actor(AdvantageEstimatorRegistry._actor_name)
        assert actor is not None

    try:
        import ray

        # 1. Register a function in the fixture's Ray session
        _register_func_and_verify()

        # 2. Force-kill the named actor before shutting down Ray. Waiting for
        # owner-death cleanup can take Ray's full graceful actor timeout.
        ray.kill(ray.get_actor(AdvantageEstimatorRegistry._actor_name))
        ray.shutdown()

        AdvantageEstimatorRegistry.unregister("named_actor_test")
        AdvantageEstimatorRegistry.shutdown_actor()

        # 3. Initialize Ray and register the function against a fresh actor.
        ray.init()
        _register_func_and_verify()

    finally:
        _remove_registry_entries(AdvantageEstimatorRegistry, "named_actor_test")
        ray.shutdown()


# ---------------------------------------------------------------------------
# compute_tis_diagnostics — the shared TIS importance-ratio diagnostics used by
# both the FSDP (PolicyWorkerBase.training_step) and Megatron
# (MegatronModelWrapper.forward_backward_mini_batch) backends.
# ---------------------------------------------------------------------------


def test_tis_diagnostics_on_policy_is_exact():
    """Identical old/rollout logprobs => ratio exactly 1.0, zero abs log-ratio."""
    lp = torch.tensor([[-0.5, -1.0, -2.0]])
    mask = torch.ones_like(lp)
    out = compute_tis_diagnostics(lp, lp.clone(), mask, cap=2.0)
    assert out == {
        "tis/imp_ratio_mean": 1.0,
        "tis/imp_ratio_capped_fraction": 0.0,
        "tis/log_ratio_abs_mean": 0.0,
    }


def test_tis_diagnostics_hand_computed_masked_means():
    """Mask-weighted means over a hand-computed case; masked tokens must not count.

    Two valid tokens with ratios 2 and 0.5 (deltas +/-log 2) and one masked token
    with a huge delta that would dominate every metric if the mask leaked.
    """
    log2 = math.log(2.0)
    old_lp = torch.tensor([[log2, -log2, 100.0]])
    rollout_lp = torch.tensor([[0.0, 0.0, -100.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    out = compute_tis_diagnostics(old_lp, rollout_lp, mask, cap=1.5)
    assert out["tis/imp_ratio_mean"] == pytest.approx((2.0 + 0.5) / 2)
    # Only the ratio-2 token exceeds cap=1.5.
    assert out["tis/imp_ratio_capped_fraction"] == pytest.approx(0.5)
    assert out["tis/log_ratio_abs_mean"] == pytest.approx(log2)


def test_tis_diagnostics_clamps_ratio_but_not_log_ratio():
    """delta=60 exponentiates at the +/-20 clamp; the abs log-ratio stays unclamped."""
    old_lp = torch.tensor([[30.0]])
    rollout_lp = torch.tensor([[-30.0]])
    mask = torch.ones_like(old_lp)
    out = compute_tis_diagnostics(old_lp, rollout_lp, mask, cap=2.0)
    assert out["tis/imp_ratio_mean"] == pytest.approx(math.exp(20.0), rel=1e-6)
    assert out["tis/log_ratio_abs_mean"] == pytest.approx(60.0)
    assert out["tis/imp_ratio_capped_fraction"] == pytest.approx(1.0)


def test_tis_diagnostics_none_rollout_keyset_identical_fallback():
    """Absent rollout logprobs must still emit the full keyset (all_reduce safety)."""
    old_lp = torch.tensor([[0.1, 0.2]])
    mask = torch.ones_like(old_lp)
    out = compute_tis_diagnostics(old_lp, None, mask, cap=2.0)
    assert tuple(out.keys()) == TIS_DIAG_KEYS
    assert out == {
        "tis/imp_ratio_mean": 1.0,
        "tis/imp_ratio_capped_fraction": 0.0,
        "tis/log_ratio_abs_mean": 0.0,
    }


def test_tis_diagnostics_all_masked_batch_emits_zeros_not_nan():
    """A fully-masked micro-batch divides by the clamped denom, never NaN."""
    old_lp = torch.tensor([[1.0, 2.0]])
    rollout_lp = torch.tensor([[0.0, 0.0]])
    mask = torch.zeros_like(old_lp)
    out = compute_tis_diagnostics(old_lp, rollout_lp, mask, cap=2.0)
    assert out["tis/imp_ratio_mean"] == 0.0
    assert out["tis/imp_ratio_capped_fraction"] == 0.0
    assert out["tis/log_ratio_abs_mean"] == 0.0
