"""Contracts for skipping the policy's old-log-prob forward (old_logprobs_from_training_forward).

With one optimizer step per batch the PPO ratio is exactly one, so the training forward can supply
the old log-probs. These tests pin the eligibility gate, the objective's equality with the explicit
on-policy computation, the driver's forward dispatch, and the parity of the consume-time mismatch
diagnostics that move from the driver into the Megatron loss closure.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed
from omegaconf import OmegaConf, open_dict

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.utils.importance_ratio_diagnostics import (
    TrainingForwardMismatchMonitor,
    mismatch_ratio_metrics,
    rollout_train_prob_diff_metrics,
)
from skyrl_train.utils.loss_reduction import reduce_loss
from skyrl_train.utils.old_logprob_forward import (
    OLD_LOGPROBS_FROM_TRAINING_FORWARD_KEY,
    old_logprob_forward_skip_problems,
    skip_old_logprob_forward,
)
from skyrl_train.utils.policy_losses import LossScaling, compute_policy_objective, ppo_policy_loss
from tests.cpu.util import example_dummy_config, stub_megatron_modules

stub_megatron_modules()

from skyrl_train.workers.megatron import megatron_model_wrapper as mmw  # noqa: E402

BATCH_SIZE = 4
SEQ_LEN = 9
NUM_ACTIONS = 5


def _eligible_config():
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.train_batch_size = 2
    cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.critic.model.path = None
    cfg.trainer.dump_data_batch = False
    cfg.trainer.policy.megatron_config.check_train_eval_parity = False
    algorithm = cfg.trainer.algorithm
    algorithm[OLD_LOGPROBS_FROM_TRAINING_FORWARD_KEY] = True
    algorithm.policy_loss_type = "regular"
    algorithm.advantage_estimator = "grpo"
    algorithm.use_kl_loss = False
    algorithm.use_kl_in_reward = False
    algorithm.use_tis = False
    algorithm.ratio_diagnostics.pooled = True
    cfg.generator.sampling_params.logprobs = 0
    return cfg


def test_eligible_config_skips_the_forward():
    assert old_logprob_forward_skip_problems(_eligible_config()) == []
    assert skip_old_logprob_forward(_eligible_config())


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (f"trainer.algorithm.{OLD_LOGPROBS_FROM_TRAINING_FORWARD_KEY}", False, "is off"),
        ("trainer.strategy", "fsdp2", "must be megatron"),
        ("trainer.algorithm.policy_loss_type", "dual_clip", "must be regular"),
        ("trainer.update_epochs_per_batch", 2, "update_epochs_per_batch"),
        ("trainer.policy_mini_batch_size", 1, "one optimizer step"),
        ("trainer.critic.model.path", "critic", "critic"),
        ("trainer.algorithm.advantage_estimator", "gae", "gae"),
        ("trainer.algorithm.use_kl_loss", True, "use_kl_loss"),
        ("trainer.algorithm.use_kl_in_reward", True, "use_kl_in_reward"),
        ("trainer.algorithm.use_tis", True, "use_tis"),
        ("trainer.algorithm.distillation", {"reward_mode": "add"}, "distillation"),
        ("trainer.policy.megatron_config.check_train_eval_parity", True, "check_train_eval_parity"),
        ("trainer.dump_data_batch", True, "dump_data_batch"),
        ("trainer.algorithm.ratio_diagnostics.pooled", False, "pooled"),
    ],
)
def test_each_disqualifier_keeps_the_forward(path, value, reason):
    cfg = _eligible_config()
    with open_dict(cfg):
        OmegaConf.update(cfg, path, value, merge=False)

    problems = old_logprob_forward_skip_problems(cfg)

    assert len(problems) == 1 and reason in problems[0]
    assert not skip_old_logprob_forward(cfg)


def test_missing_knob_keeps_the_forward():
    cfg = _eligible_config()
    with open_dict(cfg):
        del cfg.trainer.algorithm[OLD_LOGPROBS_FROM_TRAINING_FORWARD_KEY]

    assert old_logprob_forward_skip_problems(cfg) == [
        f"trainer.algorithm.{OLD_LOGPROBS_FROM_TRAINING_FORWARD_KEY} is off"
    ]


def _objective_config():
    return OmegaConf.create(
        {
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": SEQ_LEN,
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "use_tis": False,
            "use_entropy_loss": False,
            "entropy_loss_coef": 0.0,
            "use_kl_loss": False,
            "kl_estimator_type": "k1",
            "kl_loss_coef": 0.0,
            "think_token_weight": 1.0,
        }
    )


def _objective(log_probs, old_log_probs, advantages, loss_mask, entropy):
    return compute_policy_objective(
        action_log_probs=log_probs,
        old_action_log_probs=old_log_probs,
        base_action_log_probs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=entropy,
        config=_objective_config(),
        policy_loss_fn=ppo_policy_loss,
        accumulation_steps=3,
        scaling=LossScaling.MEGATRON_PIPELINE,
    )


@pytest.mark.parametrize("seed", range(5))
def test_detached_old_logprobs_give_the_on_policy_objective_bitwise(seed):
    torch.manual_seed(seed)
    base = torch.randn(BATCH_SIZE, NUM_ACTIONS)
    advantages = torch.randn(BATCH_SIZE, NUM_ACTIONS)
    loss_mask = (torch.rand(BATCH_SIZE, NUM_ACTIONS) > 0.3).float()
    loss_mask[0, 0] = 1.0
    entropy = torch.rand(BATCH_SIZE, NUM_ACTIONS)

    log_probs = base.clone().requires_grad_(True)
    skipped = _objective(log_probs, log_probs.detach(), advantages, loss_mask, entropy)
    (skipped_grad,) = torch.autograd.grad(skipped.optimization_loss, log_probs)

    log_probs = base.clone().requires_grad_(True)
    explicit = _objective(log_probs, log_probs.detach().clone(), advantages, loss_mask, entropy)
    (explicit_grad,) = torch.autograd.grad(explicit.optimization_loss, log_probs)

    assert torch.equal(skipped.optimization_loss, explicit.optimization_loss)
    assert torch.equal(skipped.unscaled_loss, explicit.unscaled_loss)
    assert torch.equal(skipped_grad, explicit_grad)
    assert skipped.metrics == explicit.metrics
    assert skipped.metrics["ppo_ratio_exact_unit_fraction"] == 1.0
    assert skipped.metrics["ppo_clip_ratio"] == 0.0

    # At a unit ratio the surrogate's value is the advantage term and its gradient is the
    # on-policy policy gradient.
    assert torch.equal(skipped.policy_loss, reduce_loss(-advantages, loss_mask, "token_mean", SEQ_LEN))
    log_probs = base.clone().requires_grad_(True)
    plain = reduce_loss(-advantages * log_probs, loss_mask, "token_mean", SEQ_LEN)
    (plain_grad,) = torch.autograd.grad(plain, log_probs)
    assert torch.equal(skipped_grad, plain_grad)


class _ForwardPolicyGroup:
    """Records the driver's forward dispatches and answers them with fixed log-probs."""

    def __init__(self, action_log_probs):
        self.actor_infos = [SimpleNamespace(rank=SimpleNamespace(dp=0, dp_size=1, is_collection_dp_rank=lambda: True))]
        self._action_log_probs = action_log_probs
        self.forward_calls = 0

    def async_run_ray_method(self, dispatch_type, method_name, *args, **kwargs):
        if method_name == "forward":
            self.forward_calls += 1
            return [TrainingOutputBatch({"output": self._action_log_probs})]
        if method_name == "empty_cache":
            return []
        raise AssertionError(f"Unexpected policy method: {method_name}")


def _forward_batch():
    torch.manual_seed(0)
    batch = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 50, (BATCH_SIZE, SEQ_LEN)),
            "attention_mask": torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long),
            "loss_mask": (torch.rand(BATCH_SIZE, NUM_ACTIONS) > 0.3).long(),
            "rollout_logprobs": torch.randn(BATCH_SIZE, NUM_ACTIONS),
            "rollout_staleness": torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        }
    )
    batch.metadata = {"response_length": NUM_ACTIONS}
    return batch


def _driver(cfg, policy_group):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.policy_model = policy_group
    trainer.critic_model = None
    trainer.ref_model = None
    trainer.colocate_all = False
    trainer.all_metrics = {}
    trainer.global_step = 3
    trainer._training_metrics_enabled = True
    return trainer


def test_driver_skips_the_forward_when_eligible():
    action_log_probs = torch.randn(BATCH_SIZE, NUM_ACTIONS)
    policy_group = _ForwardPolicyGroup(action_log_probs)
    trainer = _driver(_eligible_config(), policy_group)

    with patch("skyrl_train.trainer.ray.get", side_effect=lambda refs: refs):
        batch = trainer.fwd_logprobs_values_reward(_forward_batch())

    assert policy_group.forward_calls == 0
    assert batch["action_log_probs"] is None
    assert batch["base_action_log_probs"] is None
    assert not any(key.startswith("policy/") for key in trainer.all_metrics)


def test_driver_runs_the_forward_when_ineligible():
    cfg = _eligible_config()
    cfg.trainer.algorithm[OLD_LOGPROBS_FROM_TRAINING_FORWARD_KEY] = False
    action_log_probs = torch.randn(BATCH_SIZE, NUM_ACTIONS)
    policy_group = _ForwardPolicyGroup(action_log_probs)
    trainer = _driver(cfg, policy_group)
    batch = _forward_batch()
    expected = mismatch_ratio_metrics(
        action_log_probs, batch["rollout_logprobs"], batch["loss_mask"], batch["rollout_staleness"]
    )
    expected.update(rollout_train_prob_diff_metrics(action_log_probs, batch["rollout_logprobs"], batch["loss_mask"]))

    with patch("skyrl_train.trainer.ray.get", side_effect=lambda refs: refs):
        batch = trainer.fwd_logprobs_values_reward(batch)

    assert policy_group.forward_calls == 1
    assert torch.equal(batch["action_log_probs"], action_log_probs)
    assert trainer.all_metrics == expected


def test_training_forward_monitor_matches_the_driver_computation():
    torch.manual_seed(1)
    rows = 6
    learner = torch.randn(rows, NUM_ACTIONS)
    rollout = learner + 0.1 * torch.randn(rows, NUM_ACTIONS)
    loss_mask = (torch.rand(rows, NUM_ACTIONS) > 0.3).float()
    staleness = torch.tensor([0, 0, 1, 3, 5, 9], dtype=torch.int32)
    expected = mismatch_ratio_metrics(
        learner, rollout, loss_mask, staleness, eps_clip_low=0.2, eps_clip_high=0.05, key_prefix="mismatch/"
    )
    expected.update(rollout_train_prob_diff_metrics(learner, rollout, loss_mask, key_prefix=""))

    monitor = TrainingForwardMismatchMonitor()
    for start in range(0, rows, 2):
        end = start + 2
        monitor.add(learner[start:end], rollout[start:end], loss_mask[start:end], staleness[start:end])

    metrics = monitor.metrics(
        gather_fn=lambda tensor: tensor,
        eps_clip_low=0.2,
        eps_clip_high=0.05,
        position_window=256,
        mismatch=True,
        prob_diff=True,
    )

    assert metrics == expected
    assert "mismatch/staleness1/log_ratio_abs_mean" in metrics
    assert "rollout_train_prob_diff_std" in metrics


class _FakeMegatronModule:
    def __call__(self, sequences, position_ids, attention_mask, packed_seq_params=None, fp32_output=False):
        return torch.zeros(1)


def _fake_forward_backward_func(
    forward_step_func, data_iterator, model, num_microbatches, seq_length, micro_batch_size, forward_only
):
    metrics_list = []
    for _ in range(num_microbatches):
        outputs, closure = forward_step_func(data_iterator, model[0])
        loss, metrics = closure(outputs)
        metrics["closure_loss"] = loss.detach().clone()
        metrics_list.append(metrics)
    return metrics_list


def _wrapper_config():
    return OmegaConf.create(
        {
            "trainer": {
                "use_sample_packing": False,
                "training_metrics": True,
                "algorithm": {
                    **OmegaConf.to_container(_objective_config()),
                    "ratio_diagnostics": {"pooled": True, "position_window": 256, "exact_quantiles": False},
                },
            },
            "generator": {"sampling_params": {"temperature": 1.0, "logprobs": 0}},
        }
    )


def _megatron_metrics(monkeypatch, *, token_logprobs, old_logprobs, rollout_logprobs, loss_mask, staleness):
    """Drive the Megatron loss closure over two micro-batches with the given (or absent) old log-probs."""
    wrapper = mmw.MegatronModelWrapper.__new__(mmw.MegatronModelWrapper)
    wrapper.cfg = _wrapper_config()
    wrapper.actor_module = [_FakeMegatronModule()]
    wrapper.actor_optimizer = None
    wrapper.policy_loss_fn = ppo_policy_loss
    wrapper.use_sample_packing = False
    wrapper._logprob_chunk_size = None
    monkeypatch.setattr(mmw, "get_forward_backward_func", lambda: _fake_forward_backward_func)
    for name in ("is_pipeline_first_stage", "is_pipeline_last_stage"):
        monkeypatch.setattr(mmw.mpu, name, lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_pipeline_model_parallel_last_rank", lambda: 0, raising=False)
    world = torch.distributed.group.WORLD
    monkeypatch.setattr(mmw.mpu, "get_pipeline_model_parallel_group", lambda: world, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_data_parallel_group", lambda **kwargs: world, raising=False)

    def fake_token_logprobs(self, logits, sequences, mask, packed_seq_params):
        micro = int(sequences[0, 0].item())
        logprobs = torch.zeros(2, SEQ_LEN)
        logprobs[:, -NUM_ACTIONS:] = token_logprobs[2 * micro : 2 * micro + 2]
        return logprobs.requires_grad_(True)

    monkeypatch.setattr(mmw.MegatronModelWrapper, "_token_logprobs", fake_token_logprobs)
    monkeypatch.setattr(mmw.MegatronModelWrapper, "_token_entropies", lambda self, *args: torch.zeros(2, SEQ_LEN))

    def micro_batch(micro: int) -> mmw.MegatronPolicyMicroBatch:
        rows = slice(2 * micro, 2 * micro + 2)
        return mmw.MegatronPolicyMicroBatch(
            sequences=torch.full((2, SEQ_LEN), micro),
            attention_mask=torch.ones(2, SEQ_LEN),
            position_ids=torch.arange(SEQ_LEN).repeat(2, 1),
            num_actions=NUM_ACTIONS,
            old_action_log_probs=None if old_logprobs is None else old_logprobs[rows],
            base_action_log_probs=None,
            advantages=torch.linspace(-1, 1, 2 * NUM_ACTIONS).reshape(2, NUM_ACTIONS) * (micro + 1),
            loss_mask=loss_mask[rows],
            rollout_action_logprobs=rollout_logprobs[rows],
            response_span_tags=None,
            global_loss_denom=None,
            rollout_staleness=staleness[rows],
        )

    return wrapper.forward_backward_mini_batch(
        micro_batches=[micro_batch(0), micro_batch(1)], seq_len=SEQ_LEN, micro_batch_size=2, temperature=1.0
    )


def test_megatron_loss_takes_old_logprobs_from_its_forward(single_rank_group, monkeypatch):
    torch.manual_seed(2)
    token_logprobs = -torch.rand(BATCH_SIZE, NUM_ACTIONS)
    rollout_logprobs = token_logprobs + 0.2 * torch.randn(BATCH_SIZE, NUM_ACTIONS)
    loss_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]).float()
    staleness = torch.tensor([0, 2, 0, 5], dtype=torch.int32)

    skipped = _megatron_metrics(
        monkeypatch,
        token_logprobs=token_logprobs,
        old_logprobs=None,
        rollout_logprobs=rollout_logprobs,
        loss_mask=loss_mask,
        staleness=staleness,
    )
    explicit = _megatron_metrics(
        monkeypatch,
        token_logprobs=token_logprobs,
        old_logprobs=token_logprobs.clone(),
        rollout_logprobs=rollout_logprobs,
        loss_mask=loss_mask,
        staleness=staleness,
    )

    for skipped_metrics, explicit_metrics in zip(skipped, explicit, strict=True):
        assert torch.equal(skipped_metrics.pop("closure_loss"), explicit_metrics.pop("closure_loss"))
        assert skipped_metrics["ppo_ratio_exact_unit_fraction"] == 1.0
        assert {key: skipped_metrics[key] for key in explicit_metrics} == explicit_metrics
    assert skipped[-1]["log_ratio_abs_mean"] == 0.0
    assert skipped[-1]["log_ratio_statistics_valid"] == 1.0
    expected = mismatch_ratio_metrics(
        token_logprobs,
        rollout_logprobs,
        loss_mask,
        staleness,
        eps_clip_low=0.2,
        eps_clip_high=0.2,
        key_prefix="mismatch/",
    )
    expected.update(rollout_train_prob_diff_metrics(token_logprobs, rollout_logprobs, loss_mask, key_prefix=""))
    assert {key: skipped[-1][key] for key in expected} == expected
    assert not any(key.startswith("mismatch/") for key in explicit[-1])
    assert not any(key.startswith("mismatch/") for key in skipped[0])
