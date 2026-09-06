"""Backend-parity contracts for policy objectives and diagnostics.

These tests pin the ordinary/FSDP training-step status and Megatron pipeline
metrics for TIS, clipping, think-token weighting, global normalization, and
cross-microbatch log-ratio diagnostics.

`megatron_model_wrapper` imports `megatron.core` submodules at module load, but
nothing under test touches them, so when megatron is not installed (the CPU CI
env) we stub those submodules via the shared tests/cpu/util.py helper — only if
megatron is genuinely absent, so a real-megatron env is left untouched.
"""

import pytest
import torch
import math
from omegaconf import OmegaConf

from skyrl_train.utils.importance_ratio_diagnostics import (
    TIS_DIAG_KEYS,
    LogRatioMonitor,
    compute_tis_diagnostics,
    behavior_drift_metrics,
)
from skyrl_train.utils.policy_losses import POLICY_CLIP_METRIC_KEYS, ppo_policy_loss
from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.dataset.replay_buffer import Experience  # noqa: E402
from skyrl_train.workers.megatron import megatron_model_wrapper as mmw  # noqa: E402
from skyrl_train.workers.worker import PolicyWorkerBase  # noqa: E402

BATCH_SIZE = 2
SEQ_LEN = 6
NUM_ACTIONS = 4
CAP = 2.0


def test_behavior_drift_pools_tokens_and_keeps_ratio_direction():
    # Three tokens in unequal-length rows: ratios 1, 2, 4. A row mean is wrong.
    learner = torch.tensor([[-4.0, -4.0 + math.log(2)], [-4.0 + math.log(4), float("nan")]], dtype=torch.float64)
    behavior = torch.full_like(learner, -4.0)
    mask = torch.tensor([[1, 1], [1, 0]])
    metrics = behavior_drift_metrics(learner, behavior, mask, eps_clip_low=0.2, eps_clip_high=0.2)
    prefix = "policy/behavior_drift/"
    assert metrics[prefix + "selected_tokens"] == metrics[prefix + "finite_tokens"] == 3
    assert metrics[prefix + "log_ratio_mean"] == pytest.approx(math.log(2))
    assert metrics[prefix + "mean_squared_log_ratio"] == pytest.approx(5 * math.log(2) ** 2 / 3)
    assert metrics[prefix + "log_mean_ratio"] == pytest.approx(math.log(7 / 3))
    assert metrics[prefix + "token_weight_ess_fraction"] == pytest.approx(7 / 9)
    assert metrics[prefix + "upper_clip_pressure"] == pytest.approx(2 / 3)
    assert metrics[prefix + "lower_clip_pressure"] == 0
    reverse = behavior_drift_metrics(behavior, learner, mask, eps_clip_low=0.2, eps_clip_high=0.2)
    assert reverse[prefix + "log_ratio_mean"] == pytest.approx(-math.log(2))
    assert reverse[prefix + "lower_clip_pressure"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("behavior", [None, torch.tensor([[float("inf"), float("nan")]])])
def test_behavior_drift_missing_or_nonfinite_reports_coverage_without_fake_statistics(behavior):
    metrics = behavior_drift_metrics(torch.zeros(1, 2), behavior, torch.ones(1, 2), eps_clip_low=0.2, eps_clip_high=0.2)
    assert metrics["policy/behavior_drift/selected_tokens"] == 2
    assert metrics["policy/behavior_drift/finite_fraction"] == 0
    assert "policy/behavior_drift/token_weight_ess_fraction" not in metrics


def test_behavior_drift_preserves_extreme_tails_and_empty_selection():
    learner = torch.tensor([[-1000.0, 0.0, float("nan")]])
    behavior = torch.tensor([[0.0, -1000.0, 0.0]])
    metrics = behavior_drift_metrics(learner, behavior, torch.ones(1, 3), eps_clip_low=0.2, eps_clip_high=0.2)
    assert metrics["policy/behavior_drift/finite_fraction"] == pytest.approx(2 / 3)
    assert metrics["policy/behavior_drift/abs_log_ratio_p99"] == 1000
    assert metrics["policy/behavior_drift/mean_squared_log_ratio"] == 1000000
    assert metrics["policy/behavior_drift/token_weight_ess_fraction"] == 0.5
    empty = behavior_drift_metrics(learner, behavior, torch.zeros(1, 3), eps_clip_low=0.2, eps_clip_high=0.2)
    assert empty["policy/behavior_drift/selected_tokens"] == 0
    assert "policy/behavior_drift/abs_log_ratio_mean" not in empty


def test_behavior_drift_concentration_does_not_hide_uniform_shift():
    metrics = behavior_drift_metrics(
        torch.full((1, 3), -5.0), torch.zeros(1, 3), torch.ones(1, 3), eps_clip_low=0.2, eps_clip_high=0.2
    )
    assert metrics["policy/behavior_drift/token_weight_ess_fraction"] == 1
    assert metrics["policy/behavior_drift/log_ratio_mean"] == -5
    assert metrics["policy/behavior_drift/mean_squared_log_ratio"] == 25


def _algorithm_cfg(use_tis: bool, policy_loss_type: str = "regular") -> OmegaConf:
    return OmegaConf.create(
        {
            "trainer": {
                "use_sample_packing": False,
                "algorithm": {
                    "use_tis": use_tis,
                    "tis_imp_ratio_cap": CAP,
                    "use_entropy_loss": False,
                    "entropy_loss_coef": 0.0,
                    "use_kl_loss": False,
                    "kl_estimator_type": "k1",
                    "kl_loss_coef": 0.0,
                    "loss_reduction": "token_mean",
                    "think_token_weight": 1.0,
                    "policy_loss_type": policy_loss_type,
                    "eps_clip_low": 0.2,
                    "eps_clip_high": 0.05,
                    "max_seq_len": SEQ_LEN,
                },
            },
            "generator": {"sampling_params": {"temperature": 1.0}},
        }
    )


def _tis_tensors():
    torch.manual_seed(0)
    old_lp = torch.randn(BATCH_SIZE, NUM_ACTIONS)
    rollout_lp = old_lp + 0.3 * torch.randn(BATCH_SIZE, NUM_ACTIONS)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    return old_lp, rollout_lp, loss_mask


def _fake_policy_loss_fn(
    log_probs,
    old_log_probs,
    advantages,
    config=None,
    loss_mask=None,
    rollout_logprobs=None,
    global_loss_denom=None,
):
    del global_loss_denom
    return torch.tensor(0.25), {}


def _mask_sum_policy_loss_fn(
    log_probs,
    old_log_probs,
    advantages,
    config=None,
    loss_mask=None,
    rollout_logprobs=None,
    global_loss_denom=None,
):
    del log_probs, old_log_probs, advantages, config, rollout_logprobs, global_loss_denom
    return loss_mask.sum(), {}


def _globally_normalized_mask_sum(
    log_probs,
    old_log_probs,
    advantages,
    config=None,
    loss_mask=None,
    rollout_logprobs=None,
    global_loss_denom=None,
):
    del log_probs, old_log_probs, advantages, config, rollout_logprobs
    return loss_mask.sum() / global_loss_denom, {}


# ---------------------------------------------------------------------------
# FSDP path: PolicyWorkerBase.training_step
# ---------------------------------------------------------------------------


class _FakeHFModel:
    """Callable standing in for the HF actor: returns (action_log_probs, output)."""

    def __init__(self, action_log_probs: torch.Tensor):
        self._action_log_probs = action_log_probs

    def train(self):
        pass

    def __call__(self, sequences, num_actions, **kwargs):
        entropy = torch.zeros(sequences.shape[0], sequences.shape[1])
        return self._action_log_probs, {"entropy": entropy}


class _FakeStrategy:
    def backward(self, loss, model, optimizer):
        pass


class _FakeScheduler:
    def get_last_lr(self):
        return [1e-6]


def _fsdp_training_step_status(use_tis: bool, monkeypatch, policy_loss_type: str = "regular") -> dict:
    old_lp, rollout_lp, loss_mask = _tis_tensors()
    worker = object.__new__(PolicyWorkerBase)
    worker.cfg = _algorithm_cfg(use_tis, policy_loss_type)
    worker.model = _FakeHFModel(action_log_probs=old_lp + 0.01)
    worker.policy_loss_fn = _fake_policy_loss_fn
    worker.strategy = _FakeStrategy()
    worker.optimizer = None
    worker.scheduler = _FakeScheduler()
    worker.record_memory = False
    worker._grug_query_bias_window = None

    experience = Experience(
        sequences=torch.randint(0, 100, (BATCH_SIZE, SEQ_LEN)),
        action_log_probs=old_lp,
        base_action_log_probs=None,
        values=None,
        returns=None,
        advantages=torch.zeros(BATCH_SIZE, NUM_ACTIONS),
        attention_mask=torch.ones(BATCH_SIZE, SEQ_LEN),
        loss_mask=loss_mask,
        action_mask=torch.ones(BATCH_SIZE, NUM_ACTIONS),
        num_actions=NUM_ACTIONS,
        rollout_logprobs=rollout_lp,
        info={},
    )

    # CPU test: the step opens with experience.to_device(torch.cuda.current_device()).
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    # local_step + 1 < accumulation_steps => the optimizer-step branch is skipped.
    return worker.training_step(experience, global_step=0, local_step=0, accumulation_steps=2)


def test_fsdp_training_step_emits_tis_diagnostics(monkeypatch):
    old_lp, rollout_lp, loss_mask = _tis_tensors()
    expected = compute_tis_diagnostics(old_lp, rollout_lp, loss_mask, cap=CAP)
    status = _fsdp_training_step_status(use_tis=True, monkeypatch=monkeypatch)
    for key in TIS_DIAG_KEYS:
        assert status[key] == pytest.approx(expected[key])


def test_fsdp_training_step_no_tis_keys_when_disabled(monkeypatch):
    status = _fsdp_training_step_status(use_tis=False, monkeypatch=monkeypatch)
    assert not any(key.startswith("tis/") for key in status)


def test_fsdp_behavior_clip_keeps_rollout_divergence_diagnostics(monkeypatch):
    old_lp, rollout_lp, loss_mask = _tis_tensors()
    expected = compute_tis_diagnostics(old_lp, rollout_lp, loss_mask, cap=CAP)
    status = _fsdp_training_step_status(
        use_tis=False,
        policy_loss_type="behavior_clip",
        monkeypatch=monkeypatch,
    )

    assert "tis/imp_ratio_capped_fraction" not in status
    for key in ("tis/imp_ratio_mean", "tis/log_ratio_abs_mean"):
        assert status[key] == pytest.approx(expected[key])


def test_fsdp_training_step_completes_clip_metric_contract(monkeypatch):
    status = _fsdp_training_step_status(use_tis=False, monkeypatch=monkeypatch)
    assert {key: status[key] for key in POLICY_CLIP_METRIC_KEYS} == dict.fromkeys(POLICY_CLIP_METRIC_KEYS, 0.0)


# ---------------------------------------------------------------------------
# Megatron path: MegatronModelWrapper.forward_backward_mini_batch
# ---------------------------------------------------------------------------


class _FakeMegatronModule:
    """Stands in for a Megatron model chunk inside forward_step."""

    def __call__(self, sequences, position_ids, attention_mask, packed_seq_params=None, fp32_output=False):
        return torch.zeros(1)


def _fake_forward_backward_func(
    forward_step_func, data_iterator, model, num_microbatches, seq_length, micro_batch_size, forward_only
):
    """Drive forward_step + the loss closure per micro-batch, like Megatron does."""
    metrics_list = []
    for _ in range(num_microbatches):
        outputs, closure = forward_step_func(data_iterator, model[0])
        _, metrics = closure(outputs)
        metrics_list.append(metrics)
    return metrics_list


def _megatron_mini_batch_metrics(
    use_tis: bool,
    rollout_lp,
    monkeypatch,
    *,
    response_span_tags=None,
    think_token_weight: float = 1.0,
    global_loss_denom=None,
    policy_loss_fn=_fake_policy_loss_fn,
    log_ratio_offsets=(0.0, 0.0),
) -> list[dict]:
    old_lp, _, loss_mask = _tis_tensors()

    wrapper = mmw.MegatronModelWrapper.__new__(mmw.MegatronModelWrapper)
    wrapper.cfg = _algorithm_cfg(use_tis)
    wrapper.cfg.trainer.algorithm.think_token_weight = think_token_weight
    if global_loss_denom is not None:
        wrapper.cfg.trainer.algorithm.loss_reduction = "seq_mean_token_sum_norm_global"
    wrapper.actor_module = [_FakeMegatronModule()]
    wrapper.actor_optimizer = None
    wrapper.policy_loss_fn = policy_loss_fn
    wrapper.use_sample_packing = False
    wrapper._logprob_chunk_size = None

    # Bypass the megatron-core machinery that needs a real TP/PP mesh; everything
    # from the loss closure down (policy loss, entropy, kl, metrics dict) is real.
    monkeypatch.setattr(mmw, "get_forward_backward_func", lambda: _fake_forward_backward_func)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_first_stage", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_last_stage", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)

    def token_logprobs_for_microbatch(self, logits, seqs, mask, psp):
        del self, logits, mask, psp
        token_logprobs = torch.zeros(BATCH_SIZE, SEQ_LEN)
        token_logprobs[:, -NUM_ACTIONS:] = old_lp + seqs[:, :1].float()
        return token_logprobs

    monkeypatch.setattr(
        mmw.MegatronModelWrapper,
        "_token_logprobs",
        token_logprobs_for_microbatch,
    )
    monkeypatch.setattr(
        mmw.MegatronModelWrapper,
        "_token_entropies",
        lambda self, logits, mask, psp: torch.zeros(BATCH_SIZE, SEQ_LEN),
    )
    # World-size-1 pipeline group so the final broadcast_object_list is a no-op.
    monkeypatch.setattr(mmw.mpu, "get_pipeline_model_parallel_last_rank", lambda: 0, raising=False)
    monkeypatch.setattr(
        mmw.mpu, "get_pipeline_model_parallel_group", lambda: torch.distributed.group.WORLD, raising=False
    )

    def micro_batch(offset: float) -> mmw.MegatronPolicyMicroBatch:
        sequences = torch.zeros(BATCH_SIZE, SEQ_LEN)
        sequences[:, 0] = offset
        return mmw.MegatronPolicyMicroBatch(
            sequences=sequences,
            attention_mask=torch.ones(BATCH_SIZE, SEQ_LEN),
            position_ids=torch.arange(SEQ_LEN).repeat(BATCH_SIZE, 1),
            num_actions=NUM_ACTIONS,
            old_action_log_probs=old_lp,
            base_action_log_probs=None,
            advantages=torch.zeros(BATCH_SIZE, NUM_ACTIONS),
            loss_mask=loss_mask,
            rollout_action_logprobs=rollout_lp,
            response_span_tags=response_span_tags,
            global_loss_denom=global_loss_denom,
        )

    return wrapper.forward_backward_mini_batch(
        micro_batches=[micro_batch(offset) for offset in log_ratio_offsets],
        seq_len=SEQ_LEN,
        micro_batch_size=BATCH_SIZE,
        temperature=1.0,
    )


# The megatron tests take the shared world-size-1 gloo `single_rank_group`
# fixture (tests/cpu/conftest.py) so broadcast_object_list runs as a no-op.


def test_megatron_mini_batch_emits_tis_diagnostics(single_rank_group, monkeypatch):
    old_lp, rollout_lp, loss_mask = _tis_tensors()
    expected = compute_tis_diagnostics(old_lp, rollout_lp, loss_mask, cap=CAP)
    metrics_list = _megatron_mini_batch_metrics(use_tis=True, rollout_lp=rollout_lp, monkeypatch=monkeypatch)
    assert len(metrics_list) == 2
    for metrics in metrics_list:
        for key in TIS_DIAG_KEYS:
            assert metrics[key] == pytest.approx(expected[key])


def test_megatron_mini_batch_fallback_keyset_when_rollout_logprobs_absent(single_rank_group, monkeypatch):
    """No rollout logprobs must still land the full keyset (all_reduce safety)."""
    metrics_list = _megatron_mini_batch_metrics(use_tis=True, rollout_lp=None, monkeypatch=monkeypatch)
    for metrics in metrics_list:
        assert metrics["tis/imp_ratio_mean"] == 1.0
        assert metrics["tis/imp_ratio_capped_fraction"] == 0.0
        assert metrics["tis/log_ratio_abs_mean"] == 0.0


def test_megatron_mini_batch_no_tis_keys_when_disabled(single_rank_group, monkeypatch):
    _, rollout_lp, _ = _tis_tensors()
    metrics_list = _megatron_mini_batch_metrics(use_tis=False, rollout_lp=rollout_lp, monkeypatch=monkeypatch)
    for metrics in metrics_list:
        assert not any(key.startswith("tis/") for key in metrics)


def test_megatron_mini_batch_completes_clip_metric_contract(single_rank_group, monkeypatch):
    _, rollout_lp, _ = _tis_tensors()
    metrics_list = _megatron_mini_batch_metrics(use_tis=False, rollout_lp=rollout_lp, monkeypatch=monkeypatch)
    for metrics in metrics_list:
        assert {key: metrics[key] for key in POLICY_CLIP_METRIC_KEYS} == dict.fromkeys(POLICY_CLIP_METRIC_KEYS, 0.0)


def test_megatron_mini_batch_applies_think_weight(single_rank_group, monkeypatch):
    tags = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0]])
    metrics_list = _megatron_mini_batch_metrics(
        use_tis=False,
        rollout_lp=None,
        monkeypatch=monkeypatch,
        response_span_tags=tags,
        think_token_weight=0.25,
        policy_loss_fn=_mask_sum_policy_loss_fn,
    )

    assert [metrics["policy_loss"] for metrics in metrics_list] == pytest.approx([4.0, 4.0])


def test_megatron_mini_batch_consumes_global_loss_denominator(single_rank_group, monkeypatch):
    metrics_list = _megatron_mini_batch_metrics(
        use_tis=False,
        rollout_lp=None,
        monkeypatch=monkeypatch,
        global_loss_denom=14.0,
        policy_loss_fn=_globally_normalized_mask_sum,
    )

    assert [metrics["policy_loss"] for metrics in metrics_list] == pytest.approx([0.5, 0.5])


def test_megatron_mini_batch_emits_accumulated_log_ratio_metrics(single_rank_group, monkeypatch):
    metrics_list = _megatron_mini_batch_metrics(
        use_tis=False,
        rollout_lp=None,
        monkeypatch=monkeypatch,
        log_ratio_offsets=(0.05, 0.3),
    )

    assert metrics_list[-1]["log_ratio_abs_mean"] == pytest.approx(0.175)
    assert metrics_list[-1]["log_ratio_abs_max"] == pytest.approx(0.3)
    assert metrics_list[-1]["n_tokens_dp_gt_10pct"] == 7.0
    assert metrics_list[-1]["log_ratio_diagnostics_failed"] == 0.0


def test_megatron_mini_batch_reports_nonunit_policy_ratio(single_rank_group, monkeypatch):
    metrics_list = _megatron_mini_batch_metrics(
        use_tis=False,
        rollout_lp=None,
        monkeypatch=monkeypatch,
        policy_loss_fn=ppo_policy_loss,
        log_ratio_offsets=(0.05, 0.3),
    )

    assert metrics_list[0]["ppo_ratio_exact_unit_fraction"] == 0.0
    assert metrics_list[1]["ppo_clip_pressure_high"] == pytest.approx(1.0)


def test_megatron_mini_batch_reports_unit_policy_ratio(single_rank_group, monkeypatch):
    metrics_list = _megatron_mini_batch_metrics(
        use_tis=False,
        rollout_lp=None,
        monkeypatch=monkeypatch,
        policy_loss_fn=ppo_policy_loss,
    )

    assert [metrics["ppo_ratio_exact_unit_fraction"] for metrics in metrics_list] == [1.0, 1.0]


def test_log_ratio_monitor_marks_failed_diagnostics():
    monitor = LogRatioMonitor(torch.device("cpu"))
    monitor.add(torch.zeros(2, 3), torch.zeros(2, 2), torch.ones(2, 3))

    metrics = monitor.metrics()

    assert metrics["log_ratio_diagnostics_failed"] == 1.0
    assert metrics["log_ratio_abs_mean"] == 0.0
