"""Advantage estimators, KL estimators and controllers, policy objectives, TIS diagnostics, and validate_cfg."""

import math
from pathlib import Path
from types import SimpleNamespace

import yaml

from marinskyrl.distillation import compile_distillation_plan_from_config
from cloud.iris.launch_config import load_launch_config
from cloud.iris.tests.test_launch_config import _raw_config
from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy, GroupSelectionResult
from skyrl_train.objective.teacher import teacher_advantages

import numpy as np
import pytest
import ray
import torch
from omegaconf import DictConfig, OmegaConf

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
from skyrl_train.utils.kl_controllers import AdaptiveKLController
from skyrl_train.utils.algorithm_registry import (
    AdvantageEstimatorRegistry,
    NoGroupAdvantage,
    register_advantage_estimator,
    PolicyLossRegistry,
)
from skyrl_train.utils.utils import validate_cfg
from tests.cpu.util import example_dummy_config


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
    """A dummy config that passes validate_cfg (single-GPU placement, all batch sizes == 1)."""
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
    """Regression: seq_mean_token_sum_norm_global was implemented but rejected by validate_cfg's allow-list."""
    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.loss_reduction = loss_reduction
    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
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


def test_custom_advantage_estimator_drives_compute_advantages_and_returns():
    """Custom estimators registered through the public decorator (see examples/algorithms) are dispatched by name."""

    @register_advantage_estimator("test_custom_estimator", group_contract=NoGroupAdvantage())
    def doubled_rewards(**kwargs):
        rewards = kwargs["token_level_rewards"]
        return rewards * 2, rewards * 3

    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    try:
        adv, ret = compute_advantages_and_returns(
            token_level_rewards=rewards,
            response_mask=torch.ones_like(rewards),
            index=np.array(["0"]),
            adv_estimator="test_custom_estimator",
            config={},
        )
    finally:
        AdvantageEstimatorRegistry.unregister("test_custom_estimator")

    torch.testing.assert_close(adv, rewards * 2)
    torch.testing.assert_close(ret, rewards * 3)


def _remove_registry_entries(registry, *names: str) -> None:
    for name in names:
        if name in registry.list_available():
            registry.unregister(name)
    registry.shutdown_actor()


@pytest.mark.usefixtures("ray_module")
def test_registry_cross_ray_process():
    """Functions registered on the driver are callable from Ray workers, including ones registered after init."""
    try:

        def test_policy_loss(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
            return TokenLoss(-2 * inputs.log_probs * inputs.advantages, {})

        def test_policy_loss_2(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
            return TokenLoss(-3 * inputs.log_probs * inputs.advantages, {})

        def test_advantage_estimator(**kwargs):
            rewards = kwargs["token_level_rewards"]
            return rewards * 2, rewards * 3

        PolicyLossRegistry.register("cross_process_test", test_policy_loss, spec=LossSpec(RatioAnchor.NONE))
        AdvantageEstimatorRegistry.register(
            "cross_process_adv_test", test_advantage_estimator, group_contract=NoGroupAdvantage()
        )

        @ray.remote
        def test_ray_registry_access(name: str):
            policy_loss = PolicyLossRegistry.get(name)
            adv_estimator = AdvantageEstimatorRegistry.get("cross_process_adv_test")

            log_probs = torch.tensor([[-0.4]], requires_grad=True)
            loss = policy_loss(
                PolicyLossInputs(
                    log_probs=log_probs,
                    old_log_probs=torch.tensor([[-0.5]]),
                    rollout_log_probs=None,
                    advantages=torch.tensor([[3.0]]),
                    loss_mask=torch.ones_like(log_probs),
                ),
                DictConfig({"policy_loss_type": name}),
            )
            loss.values.sum().backward()

            adv, ret = adv_estimator(
                token_level_rewards=torch.tensor([[1.0, 2.0]]),
                response_mask=torch.tensor([[1.0, 1.0]]),
                index=np.array(["0", "0"]),
            )
            return loss.values.detach(), log_probs.grad, adv, ret

        loss, gradient, adv, ret = ray.get(test_ray_registry_access.remote("cross_process_test"))
        torch.testing.assert_close(loss, torch.tensor([[2.4]]))
        torch.testing.assert_close(gradient, torch.tensor([[-6.0]]))
        torch.testing.assert_close(adv, torch.tensor([[2.0, 4.0]]))
        torch.testing.assert_close(ret, torch.tensor([[3.0, 6.0]]))

        PolicyLossRegistry.register("cross_process_test_2", test_policy_loss_2, spec=LossSpec(RatioAnchor.NONE))
        loss_2, gradient_2, _, _ = ray.get(test_ray_registry_access.remote("cross_process_test_2"))
        torch.testing.assert_close(loss_2, torch.tensor([[3.6]]))
        torch.testing.assert_close(gradient_2, torch.tensor([[-9.0]]))
    finally:
        _remove_registry_entries(PolicyLossRegistry, "cross_process_test", "cross_process_test_2")
        _remove_registry_entries(AdvantageEstimatorRegistry, "cross_process_adv_test")


@pytest.mark.parametrize("temperature", [0.7, 1.2])
@pytest.mark.parametrize(
    ("correction", "policy_loss_type"),
    [("tis", "regular"), ("none", "behavior_clip")],
    ids=["tis", "behavior-clip"],
)
def test_validate_cfg_configures_behavior_logprob_probability_convention(temperature, correction, policy_loss_type):
    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.off_policy_correction = correction
    cfg.trainer.algorithm.policy_loss_type = policy_loss_type
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


def test_validate_cfg_best_of_n_uses_selected_batch_geometry():
    cfg = _validatable_dummy_config()
    OmegaConf.update(cfg, "trainer.trajectory_selector.type", "best_of_n", force_add=True)
    cfg.generator.n_samples_per_prompt = 4
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    validate_cfg(cfg)

    assert cfg.trainer.algorithm.resolved_group_advantage.physical_group_size == 1


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


def test_reward_estimator_broadcasts_eligible_reward_without_centering():
    rewards = torch.tensor([[1.0, torch.nan, 2.0], [-4.0, 1.0, torch.nan], [torch.nan] * 3], requires_grad=True)
    mask = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 0, 0]])
    advantages, returns = compute_advantages_and_returns(
        token_level_rewards=rewards,
        response_mask=mask,
        index=np.array(["a", "b", "b"]),
        adv_estimator="reward",
        config={},
    )
    expected = torch.tensor([[3.0, 0.0, 3.0], [-3.0, -3.0, 0.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)
    assert not advantages.requires_grad


@pytest.mark.parametrize("recipe", ["grpo", "dapo", "dr_grpo", "gspo", "cispo", "opd", "mopd"])
def test_algorithm_recipe_launch_drives_policy_value_and_gradient(tmp_path: Path, recipe: str):
    raw = _raw_config()
    raw["skyrl"]["config_groups"] = {"algorithm_recipe": recipe}
    raw["skyrl"]["generator"]["n_samples_per_prompt"] = 2
    teacher_recipe = recipe in {"opd", "mopd"}
    if teacher_recipe:
        raw["skyrl"]["trainer"]["algorithm"]["distillation"] = {"routing_plan": "expert", "coefficient": 1.0}
        raw["skyrl"]["teachers"] = {
            "expert": dict(
                source="openai_compatible",
                placement="external",
                evidence="chosen_token",
                model=dict(path="teacher", revision="teacher-revision"),
                endpoints=[dict(url="https://teacher.example/v1", max_concurrency=1)],
                tokenizer_fingerprint=f"sha256:{'a' * 64}",
                max_sequence_length=1024,
                request_timeout_seconds=30,
            )
        }
        raw["skyrl"]["teacher_routing"] = {
            "expert": dict(revision="route-revision", routes=dict(default=dict(teacher="expert", weight=1.0)))
        }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw))
    launch_config = load_launch_config(path).skyrl
    config = launch_config.trainer.algorithm
    if recipe == "dapo":
        raw["skyrl"]["trainer"]["algorithm"]["dynamic_sampling"] = {"type": None}
        path.write_text(yaml.safe_dump(raw))
        sampling = load_launch_config(path).skyrl.trainer.algorithm.dynamic_sampling
        selection = GroupSelectionPolicy(DynamicSamplingType(sampling.type) if sampling.type is not None else None)
        group = SimpleNamespace(trajectory_batch={"response_ids": [[1], [2]], "rewards": [1.0, 1.0]})
        assert selection.evaluate(group) is GroupSelectionResult.KEEP

    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    old = torch.full_like(mask, -12.0 if teacher_recipe else -2.0)
    current = (old + torch.tensor([[1.1, 1.1], [1.25, 1.25]]).log()).requires_grad_()
    if teacher_recipe:
        plan = compile_distillation_plan_from_config(launch_config)
        assert plan is not None
        advantages, _ = teacher_advantages(
            old + torch.tensor([[-9.0, 0.0], [9.0, 9.0]]),
            old,
            mask.bool(),
            torch.ones_like(mask),
            plan.advantage_clip,
        )
    else:
        advantages, _ = compute_advantages_and_returns(
            token_level_rewards=torch.tensor([[0.0, 0.0], [0.0, 2.0]]),
            response_mask=mask,
            index=np.array(["prompt", "prompt"]),
            adv_estimator=config.advantage_estimator,
            config=config,
            grpo_norm_by_std=config.grpo_norm_by_std,
        )
    batch = build_objective_micro_batch(
        action_log_probs=current,
        old_action_log_probs=old,
        base_action_log_probs=None,
        advantages=advantages,
        loss_mask=mask,
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros_like(mask),
        think_token_weight=1,
        teacher=None,
    )
    counts = step_counts([mask], [mask], [], [advantages], 8, lambda value: value)
    result = compute_policy_objective(
        batch,
        loss=PolicyLossRegistry.get(config.policy_loss_type),
        counts=counts,
        config=config,
        loss_scale=1,
        report_scale=1,
    )
    scale = {"opd": 9.0, "mopd": 5.0, "dr_grpo": 1.0}.get(recipe, 1 / (math.sqrt(2) + 1e-6))
    denominator = {"grpo": 2, "dapo": 3, "dr_grpo": 16, "gspo": 2, "cispo": 3, "opd": 3, "mopd": 2}[recipe]
    upper = {"grpo": 1.2, "dr_grpo": 1.2, "gspo": 1.0004}.get(recipe, 1.25)
    second_weight = 1.0 if recipe in {"grpo", "gspo", "mopd"} else 2.0
    expected_value = (1.1 - second_weight * upper) * scale / denominator
    if recipe == "cispo":
        expected_value = (1.1 * (-2 + math.log(1.1)) - 2.5 * (-2 + math.log(1.25))) * scale / 3
    torch.testing.assert_close(result.optimization_loss, torch.tensor(expected_value), rtol=1e-5, atol=1e-7)
    result.optimization_loss.backward()
    positive_gradient = -1.25 * scale / denominator if upper == 1.25 else 0.0
    if recipe == "mopd":
        positive_gradient /= 2
    expected_gradient = torch.tensor([[1.1 * scale / denominator, 0], [positive_gradient, positive_gradient]])
    torch.testing.assert_close(current.grad, expected_gradient, rtol=1e-5, atol=1e-7)
