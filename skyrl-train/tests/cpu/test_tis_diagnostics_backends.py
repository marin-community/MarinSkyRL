"""Both training backends must emit the TIS importance-ratio diagnostics.

The diagnostics (tis/imp_ratio_mean, tis/imp_ratio_capped_fraction,
tis/log_ratio_abs_mean) are a pure function of tensors every backend
materializes for the TIS loss, computed by the shared
`skyrl_train.utils.ppo_utils.compute_tis_diagnostics` (arithmetic covered in
tests/cpu/utils/test_ppo_utils.py). These tests pin the two call sites: the
FSDP path (`PolicyWorkerBase.training_step` status dict) and the Megatron path
(`MegatronModelWrapper.forward_backward_mini_batch` per-micro metrics dicts).
The Megatron path silently emitting NO tis/* keys is the production blind spot
that motivated the shared helper — a second inlined copy could drift or be
dropped again.

`megatron_model_wrapper` imports `megatron.core` submodules at module load, but
nothing under test touches them, so when megatron is not installed (the CPU CI
env) we stub those submodules via the shared tests/cpu/util.py helper — only if
megatron is genuinely absent, so a real-megatron env is left untouched.
"""

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.utils.ppo_utils import TIS_DIAG_KEYS, compute_tis_diagnostics
from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.dataset.replay_buffer import Experience  # noqa: E402
from skyrl_train.workers.megatron import megatron_model_wrapper as mmw  # noqa: E402
from skyrl_train.workers.worker import PolicyWorkerBase  # noqa: E402

BATCH_SIZE = 2
SEQ_LEN = 6
NUM_ACTIONS = 4
CAP = 2.0


def _algorithm_cfg(use_tis: bool) -> OmegaConf:
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


def _fake_policy_loss_fn(log_probs, old_log_probs, advantages, config=None, loss_mask=None, rollout_logprobs=None):
    return torch.tensor(0.25), 0.0


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


def _fsdp_training_step_status(use_tis: bool, monkeypatch) -> dict:
    old_lp, rollout_lp, loss_mask = _tis_tensors()
    worker = object.__new__(PolicyWorkerBase)
    worker.cfg = _algorithm_cfg(use_tis)
    worker.model = _FakeHFModel(action_log_probs=old_lp + 0.01)
    worker.policy_loss_fn = _fake_policy_loss_fn
    worker.strategy = _FakeStrategy()
    worker.optimizer = None
    worker.scheduler = _FakeScheduler()
    worker.record_memory = False

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


def _megatron_mini_batch_metrics(use_tis: bool, rollout_lp, monkeypatch) -> list[dict]:
    old_lp, _, loss_mask = _tis_tensors()

    wrapper = mmw.MegatronModelWrapper.__new__(mmw.MegatronModelWrapper)
    wrapper.cfg = _algorithm_cfg(use_tis)
    wrapper.actor_module = [_FakeMegatronModule()]
    wrapper.actor_optimizer = None
    wrapper.policy_loss_fn = _fake_policy_loss_fn
    wrapper.use_sample_packing = False
    wrapper._logprob_chunk_size = None

    # Bypass the megatron-core machinery that needs a real TP/PP mesh; everything
    # from the loss closure down (policy loss, entropy, kl, metrics dict) is real.
    monkeypatch.setattr(mmw, "get_forward_backward_func", lambda: _fake_forward_backward_func)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_first_stage", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_last_stage", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    token_logprobs = torch.zeros(BATCH_SIZE, SEQ_LEN)
    token_logprobs[:, -NUM_ACTIONS:] = old_lp
    monkeypatch.setattr(
        mmw.MegatronModelWrapper, "_token_logprobs", lambda self, logits, seqs, mask, psp: token_logprobs
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

    micro_batch = {
        "sequences": torch.randint(0, 100, (BATCH_SIZE, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH_SIZE, SEQ_LEN),
        "position_ids": torch.arange(SEQ_LEN).repeat(BATCH_SIZE, 1),
        "num_actions": NUM_ACTIONS,
        "old_action_log_probs": old_lp,
        "base_action_log_probs": None,
        "advantages": torch.zeros(BATCH_SIZE, NUM_ACTIONS),
        "loss_mask": loss_mask,
        "rollout_action_logprobs": rollout_lp,
    }
    return wrapper.forward_backward_mini_batch(
        micro_batches=[micro_batch, dict(micro_batch)],
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
