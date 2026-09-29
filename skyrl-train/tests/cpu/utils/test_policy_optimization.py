"""
Run with:
uv run --isolated --group dev --extra cpu pytest tests/cpu/utils/test_policy_optimization.py
"""

import torch
import math
import pytest
from omegaconf import OmegaConf
from skyrl_train.utils.policy_math import compute_approx_kl
from skyrl_train.objective.losses import PolicyLossInputs, TokenLoss, ppo_policy_loss
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import step_counts
from skyrl_train.config.objective_spec import LossSpec, RatioAnchor
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


@pytest.mark.parametrize("coefficient", [0.0, 0.1, 1.0])
def test_policy_objective_kl_gradient_matches_analytic_derivative(coefficient):
    log_probs = torch.tensor([[-0.8, -1.3, -0.6], [-1.3, -0.8, -0.6]], dtype=torch.float64, requires_grad=True)
    base_log_probs = torch.full_like(log_probs, -1.0)
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    config = OmegaConf.create(
        {
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 3,
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "think_token_weight": 1.0,
            "use_entropy_loss": False,
            "entropy_loss_coef": 0.0,
            "use_kl_loss": True,
            "kl_loss_coef": coefficient,
            "kl_estimator_type": "k3",
            "use_tis": False,
        }
    )
    batch = build_objective_micro_batch(
        action_log_probs=log_probs,
        old_action_log_probs=log_probs.detach(),
        base_action_log_probs=base_log_probs,
        advantages=torch.zeros_like(log_probs),  # Isolate KL in the real PPO objective.
        loss_mask=mask,
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros_like(log_probs),
        think_token_weight=1.0,
        teacher=None,
    )
    counts = step_counts([mask], [mask], [], [torch.zeros_like(mask)], 3, lambda value: value)
    objective = compute_policy_objective(
        batch, loss=ppo_policy_loss, counts=counts, config=config, loss_scale=1, report_scale=1
    )
    objective.optimization_loss.backward()

    # k3 derivatives at log(p/q) = [0.2, -0.3], away from clamps.
    derivatives = [1.0 - math.exp(-0.2), 1.0 - math.exp(0.3)]
    expected = torch.zeros_like(log_probs)
    # Two active tokens in one trainable sequence; the empty row contributes neither numerator nor count.
    expected[0, :2] = torch.tensor(derivatives, dtype=log_probs.dtype) * coefficient / 2
    torch.testing.assert_close(log_probs.grad, expected, rtol=1e-12, atol=1e-12)

    metric = compute_approx_kl(log_probs, base_log_probs, mask, kl_estimator_type="k3")
    assert not metric.requires_grad
    torch.testing.assert_close(objective.rows.kl.detach(), metric[0, :2].sum() / 2)


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
    error_type = ValueError if config_path == "trainer.algorithm.loss_reduction" else AssertionError
    with pytest.raises(error_type, match=error):
        validate_cfg(cfg)


def test_validate_cfg_requires_grug_query_bias_update_mode():
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import validate_cfg

    cfg = _validatable_dummy_config()
    del cfg.trainer.policy.grug_query_bias_update_mode

    with pytest.raises(AssertionError, match="missing required policy configuration: grug_query_bias_update_mode"):
        validate_cfg(cfg)


@pytest.mark.parametrize("weight", [None, 0.0, 1.0, float("inf"), [0.1]])
def test_validate_cfg_rejects_invalid_grug_query_bias_interpolation_weight(weight):
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import validate_cfg

    cfg = _validatable_dummy_config()
    cfg.trainer.policy.grug_query_bias_update_mode = "interpolate"
    cfg.trainer.policy.grug_query_bias_interpolation_weight = weight

    with pytest.raises(AssertionError, match="grug_query_bias_interpolation_weight"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_interpolation_weight_for_other_grug_query_bias_modes():
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import validate_cfg

    cfg = _validatable_dummy_config()
    cfg.trainer.policy.grug_query_bias_update_mode = "replace"
    cfg.trainer.policy.grug_query_bias_interpolation_weight = 0.1

    with pytest.raises(AssertionError, match="only valid"):
        validate_cfg(cfg)


@pytest.mark.parametrize("rate", [None, 0.0, -0.001, float("inf"), [0.001]])
def test_validate_cfg_rejects_invalid_grug_loss_free_update_rate(rate):
    cfg = _validatable_dummy_config()
    cfg.trainer.policy.grug_query_bias_update_mode = "loss_free"
    cfg.trainer.policy.grug_query_bias_update_rate = rate

    with pytest.raises(AssertionError, match="grug_query_bias_update_rate"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_loss_free_update_rate_for_other_modes():
    cfg = _validatable_dummy_config()
    cfg.trainer.policy.grug_query_bias_update_mode = "replace"
    cfg.trainer.policy.grug_query_bias_update_rate = 0.001

    with pytest.raises(AssertionError, match="only valid"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_stacked_behavior_clip_and_tis():
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import validate_cfg

    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.policy_loss_type = "behavior_clip"
    cfg.trainer.algorithm.use_tis = True

    with pytest.raises(ValueError, match="cannot be combined with use_tis"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_gspo_without_sequence_mean_reduction():
    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.policy_loss_type = "gspo"
    cfg.trainer.algorithm.loss_reduction = "token_mean"

    with pytest.raises(ValueError, match="gspo requires trainer.algorithm.loss_reduction=sequence_mean"):
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


def test_registered_policy_loss_preserves_per_token_gradients():
    @register_policy_loss("test_policy_decorator", LossSpec(RatioAnchor.NONE))
    def decorated_policy_loss(inputs, config):
        return TokenLoss(-inputs.log_probs * inputs.advantages, {})

    try:
        log_probs = torch.tensor([[-0.1, -0.5]], requires_grad=True)
        inputs = PolicyLossInputs(
            log_probs, log_probs.detach(), None, torch.tensor([[3.0, -2.0]]), torch.ones_like(log_probs)
        )
        result = PolicyLossRegistry.get("test_policy_decorator")(inputs, OmegaConf.create({}))
        torch.testing.assert_close(result.values, torch.tensor([[0.3, -1.0]]))
        result.values.sum().backward()
        torch.testing.assert_close(log_probs.grad, torch.tensor([[-3.0, 2.0]]))
    finally:
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


def test_validate_cfg_applies_custom_loss_contract_to_training():
    def custom_policy_loss(inputs, config):
        return TokenLoss(-inputs.log_probs * inputs.advantages, {})

    PolicyLossRegistry.register(
        "custom_policy", custom_policy_loss, spec=LossSpec(RatioAnchor.NONE, sequence_level=True)
    )
    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.policy_loss_type = "custom_policy"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_pipeline_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = 1
    try:
        cfg.trainer.algorithm.loss_reduction = "token_mean"
        with pytest.raises(ValueError, match="requires trainer.algorithm.loss_reduction=sequence_mean"):
            validate_cfg(cfg)
        cfg.trainer.algorithm.loss_reduction = "sequence_mean"
        validate_cfg(cfg)
        log_probs = torch.tensor([[-0.1, -0.5]], requires_grad=True)
        advantages = torch.tensor([[3.0, -2.0]])
        mask = torch.ones_like(log_probs)
        batch = build_objective_micro_batch(
            action_log_probs=log_probs,
            old_action_log_probs=log_probs.detach(),
            base_action_log_probs=None,
            advantages=advantages,
            loss_mask=mask,
            rollout_logprobs=None,
            response_span_tags=None,
            token_entropy=torch.zeros_like(log_probs),
            think_token_weight=1,
            teacher=None,
        )
        counts = step_counts([mask], [mask], [], [advantages], 2, lambda value: value)
        result = compute_policy_objective(
            batch,
            loss=PolicyLossRegistry.get("custom_policy"),
            counts=counts,
            config=cfg.trainer.algorithm,
            loss_scale=1,
            report_scale=1,
        )
        result.optimization_loss.backward()
        torch.testing.assert_close(log_probs.grad, torch.tensor([[-1.5, 1.0]]))
    finally:
        PolicyLossRegistry.unregister("custom_policy")


def _remove_registry_entries(registry, *names: str) -> None:
    for name in names:
        if name in registry.list_available():
            registry.unregister(name)
    registry.shutdown_actor()


@pytest.mark.usefixtures("ray_module")
def test_registry_cross_ray_process():
    import ray

    def custom_loss(inputs, config):
        return TokenLoss(-2 * inputs.log_probs * inputs.advantages, {})

    @ray.remote
    def gradient_from_registered_loss():
        log_probs = torch.tensor([[-0.4]], requires_grad=True)
        inputs = PolicyLossInputs(
            log_probs, log_probs.detach(), None, torch.tensor([[3.0]]), torch.ones_like(log_probs)
        )
        result = PolicyLossRegistry.get("cross_process_test")(inputs, OmegaConf.create({}))
        result.values.sum().backward()
        return result.values.detach(), log_probs.grad

    try:
        PolicyLossRegistry.register("cross_process_test", custom_loss, spec=LossSpec(RatioAnchor.NONE))
        values, gradient = ray.get(gradient_from_registered_loss.remote())
        torch.testing.assert_close(values, torch.tensor([[2.4]]))
        torch.testing.assert_close(gradient, torch.tensor([[-6.0]]))
    finally:
        _remove_registry_entries(PolicyLossRegistry, "cross_process_test")


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
# the Megatron
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


@pytest.mark.parametrize("temperature", [0.7, 1.2])
@pytest.mark.parametrize(
    ("use_tis", "policy_loss_type"),
    [(True, "regular"), (False, "behavior_clip")],
    ids=["tis", "behavior-clip"],
)
def test_validate_cfg_configures_behavior_logprob_probability_convention(temperature, use_tis, policy_loss_type):
    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.use_tis = use_tis
    cfg.trainer.algorithm.policy_loss_type = policy_loss_type
    cfg.trainer.algorithm.tis_imp_ratio_cap = 2.0
    cfg.generator.sampling_params.temperature = temperature
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    cfg.generator.num_inference_engines = 1
    validate_cfg(cfg)

    assert cfg.generator.sampling_params.logprobs == 0
    assert cfg.generator.engine_init_kwargs.logprobs_mode == "processed_logprobs"
    assert cfg.generator.engine_init_kwargs.generation_config == "vllm"
    assert cfg.generator.engine_init_kwargs.validate_rollout_logprob_sampling is True
    assert cfg.generator.sampling_params.min_tokens == 0


def test_validate_cfg_rejects_raw_tis_logprobs():
    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.use_tis = True
    cfg.trainer.algorithm.tis_imp_ratio_cap = 2.0
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    cfg.generator.num_inference_engines = 1
    OmegaConf.update(cfg.generator.engine_init_kwargs, "logprobs_mode", "raw_logprobs", force_add=True)

    with pytest.raises(ValueError, match="processed rollout logprobs"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_behavior_clip_top_p_filter():
    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.use_tis = False
    cfg.trainer.algorithm.policy_loss_type = "behavior_clip"
    cfg.generator.sampling_params.top_p = 0.95

    with pytest.raises(ValueError, match="top_p=0.95"):
        validate_cfg(cfg)


def test_validate_cfg_best_of_n_uses_selected_batch_geometry():
    cfg = _validatable_dummy_config()
    OmegaConf.update(cfg, "trainer.trajectory_selector.type", "best_of_n", force_add=True)
    cfg.generator.n_samples_per_prompt = 4
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    validate_cfg(cfg)

    assert cfg.trainer.algorithm.resolved_group_advantage.physical_group_size == 1


def test_validate_cfg_rejects_best_of_n_with_group_relative_advantages():
    cfg = _validatable_dummy_config()
    OmegaConf.update(cfg, "trainer.trajectory_selector.type", "best_of_n", force_add=True)
    cfg.generator.n_samples_per_prompt = 4
    cfg.trainer.algorithm.advantage_estimator = "grpo"

    with pytest.raises(ValueError, match="no-group advantage"):
        validate_cfg(cfg)
