"""Trainer side of trainer.algorithm.context_distillation: the rollout-context tensors, re-basing the behaviour
logprobs so TIS keeps correcting engine-vs-trainer mismatch only, the shift metrics, and the logprob phase's
second forward under the served prompt."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import skyrl_train.trainer as trainer_module
import torch
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.trajectory_runners.context_distillation import (
    CONTEXT_EDITED_KEY,
    ROLLOUT_PROMPT_TOKEN_IDS_KEY,
    ContextDistillationConfig,
)
from skyrl_train.utils.context_distillation import (
    CONTEXT_EDITED_TENSOR_KEY,
    CONTEXT_SHIFT_METRIC_KEYS,
    EDITED_FRACTION_METRIC,
    ROLLOUT_ATTENTION_MASK_KEY,
    ROLLOUT_CONTEXT_TENSOR_KEYS,
    ROLLOUT_SEQUENCES_KEY,
    SHIFT_ABS_PER_TOKEN_METRIC,
    SHIFT_PER_TOKEN_METRIC,
    SHIFT_PER_TRAJECTORY_METRIC,
    apply_context_distillation_references,
    context_shift_metrics,
    neutralize_behavior_logprobs,
    rebase_behavior_logprobs,
    rollout_context_forward_batch,
)

from tests.cpu.util import example_dummy_config


def _tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer


def _trajectory_batch(with_rollout_context=True):
    batch = {
        "prompt_token_ids": [[1, 2], [3]],
        "response_ids": [[4, 5, 6], [7]],
        "rewards": [[0.0, 0.0, 1.0], [0.0]],
        "loss_masks": [[1, 1, 1], [1]],
        "rollout_logprobs": [[-0.1, -0.2, -0.3], [-0.4]],
    }
    if with_rollout_context:
        batch[ROLLOUT_PROMPT_TOKEN_IDS_KEY] = [[1, 2, 8, 9], [3]]
        batch[CONTEXT_EDITED_KEY] = [True, False]
    return batch


def _conversion_trainer():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = example_dummy_config()
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    trainer.tokenizer = _tokenizer()
    trainer.pad_batch = lambda batch: batch
    return trainer


# ---------------------------------------------------------------------------
# convert_to_training_input / pad_batch
# ---------------------------------------------------------------------------
def test_convert_to_training_input_attaches_the_rollout_context():
    batch = _conversion_trainer().convert_to_training_input(_trajectory_batch(), ["a", "b"])
    # served prompts are padded on their own axis; the response axis is shared with `sequences`
    assert batch[ROLLOUT_SEQUENCES_KEY].tolist() == [[1, 2, 8, 9, 4, 5, 6], [0, 0, 0, 3, 7, 0, 0]]
    assert batch[ROLLOUT_ATTENTION_MASK_KEY].tolist() == [[1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 0]]
    assert batch["sequences"].tolist() == [[1, 2, 4, 5, 6], [0, 3, 7, 0, 0]]
    torch.testing.assert_close(batch[ROLLOUT_SEQUENCES_KEY][:, -3:], batch["sequences"][:, -3:])
    assert batch[CONTEXT_EDITED_TENSOR_KEY].dtype == torch.bool
    assert batch[CONTEXT_EDITED_TENSOR_KEY].tolist() == [True, False]
    assert batch.metadata["response_length"] == 3


def test_convert_to_training_input_without_rollout_context_keeps_the_key_set():
    batch = _conversion_trainer().convert_to_training_input(_trajectory_batch(with_rollout_context=False), ["a", "b"])
    assert not any(key in batch for key in ROLLOUT_CONTEXT_TENSOR_KEYS)


def test_pad_batch_marks_padded_rows_unedited():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=2))])
    trainer.critic_model = None
    trainer.ref_model = None
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 3]]),
            "loss_mask": torch.tensor([[1, 1]]),
            CONTEXT_EDITED_TENSOR_KEY: torch.tensor([True]),
        }
    )
    batch.metadata = {"uids": ["a"]}
    padded = trainer.pad_batch(batch)
    assert padded[CONTEXT_EDITED_TENSOR_KEY].tolist() == [True, False]
    assert padded["loss_mask"].tolist() == [[1, 1], [0, 0]]
    assert padded["sequences"].tolist() == [[1, 2, 3], [1, 2, 3]]


# ---------------------------------------------------------------------------
# Behaviour-logprob re-basing and the shift metrics
# ---------------------------------------------------------------------------
ENGINE = torch.tensor([[-1.0, -2.0], [-1.5, -0.5]])  # sampled under the rollout context
TRAINING = torch.tensor([[-1.2, -2.5], [-1.4, -0.6]])  # log pi(y | training context)
ROLLOUT = torch.tensor([[-0.9, -2.1], [-1.6, -0.4]])  # log pi(y | rollout context)
EDITED = torch.tensor([True, False])


def test_rebase_leaves_only_the_engine_mismatch_in_the_ratio():
    rebased = rebase_behavior_logprobs(ENGINE, TRAINING, ROLLOUT, EDITED)
    torch.testing.assert_close(torch.exp(TRAINING[0] - rebased[0]), torch.exp(ROLLOUT[0] - ENGINE[0]))
    torch.testing.assert_close(rebased[1], ENGINE[1])


def test_neutralize_gives_edited_rows_a_ratio_of_one():
    neutral = neutralize_behavior_logprobs(ENGINE, TRAINING, EDITED)
    torch.testing.assert_close(torch.exp(TRAINING[0] - neutral[0]), torch.ones(2))
    torch.testing.assert_close(neutral[1], ENGINE[1])


def test_shift_metrics_are_hand_computable():
    training = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -1.0, -1.0]])
    rollout = torch.tensor([[-0.5, -2.0, -2.0], [-0.2, -0.2, -0.2]])
    loss_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    metrics = context_shift_metrics(training, rollout, loss_mask, torch.tensor([True, False]))
    # row 0 only, masked tokens: shifts +0.5 and 0.0
    assert metrics[EDITED_FRACTION_METRIC] == pytest.approx(0.5)
    assert metrics[SHIFT_PER_TOKEN_METRIC] == pytest.approx(0.25)
    assert metrics[SHIFT_ABS_PER_TOKEN_METRIC] == pytest.approx(0.25)
    assert metrics[SHIFT_PER_TRAJECTORY_METRIC] == pytest.approx(0.5)


def test_shift_metrics_keep_the_key_set_without_edited_rows():
    metrics = context_shift_metrics(TRAINING, ROLLOUT, torch.ones(2, 2), torch.tensor([False, False]))
    assert set(metrics) == set(CONTEXT_SHIFT_METRIC_KEYS)
    assert all(value == 0.0 for value in metrics.values())


def _logprob_phase_batch(edited=(True, False), with_rollout_logprobs=True):
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 4, 5], [0, 3, 7, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [0, 1, 1, 0]]),
            "loss_mask": torch.tensor([[1, 1], [1, 0]]),
            "rollout_logprobs": ENGINE.clone() if with_rollout_logprobs else None,
            ROLLOUT_SEQUENCES_KEY: torch.tensor([[1, 2, 8, 4, 5], [0, 0, 3, 7, 0]]),
            ROLLOUT_ATTENTION_MASK_KEY: torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 0]]),
            CONTEXT_EDITED_TENSOR_KEY: torch.tensor(list(edited)),
        }
    )
    batch.metadata = {"response_length": 2, "uids": ["a", "b"]}
    return batch


def test_apply_references_rebases_reports_and_drops_the_rollout_tensors():
    batch = _logprob_phase_batch()
    metrics = apply_context_distillation_references(
        batch,
        training_context_logprobs=TRAINING,
        rollout_context_logprobs=ROLLOUT,
        config=ContextDistillationConfig(enabled=True),
    )
    assert not any(key in batch for key in ROLLOUT_CONTEXT_TENSOR_KEYS)
    torch.testing.assert_close(batch["rollout_logprobs"], rebase_behavior_logprobs(ENGINE, TRAINING, ROLLOUT, EDITED))
    assert set(metrics) == set(CONTEXT_SHIFT_METRIC_KEYS)
    assert metrics[EDITED_FRACTION_METRIC] == pytest.approx(0.5)


def test_apply_references_neutralizes_without_the_second_forward():
    batch = _logprob_phase_batch()
    metrics = apply_context_distillation_references(
        batch,
        training_context_logprobs=TRAINING,
        rollout_context_logprobs=None,
        config=ContextDistillationConfig(enabled=True, tis_reference="none"),
    )
    torch.testing.assert_close(batch["rollout_logprobs"], neutralize_behavior_logprobs(ENGINE, TRAINING, EDITED))
    assert metrics[EDITED_FRACTION_METRIC] == pytest.approx(0.5)
    assert metrics[SHIFT_PER_TOKEN_METRIC] == 0.0


def test_apply_references_needs_the_second_forward_for_the_rollout_reference():
    with pytest.raises(ValueError, match="rollout-context"):
        apply_context_distillation_references(
            _logprob_phase_batch(),
            training_context_logprobs=TRAINING,
            rollout_context_logprobs=None,
            config=ContextDistillationConfig(enabled=True),
        )


def test_apply_references_is_a_no_op_without_rollout_context():
    batch = TrainingInputBatch({"sequences": torch.tensor([[1, 2]]), "rollout_logprobs": torch.tensor([[-1.0]])})
    assert (
        apply_context_distillation_references(
            batch,
            training_context_logprobs=torch.zeros(1, 1),
            rollout_context_logprobs=None,
            config=ContextDistillationConfig(enabled=True),
        )
        == {}
    )
    assert batch["rollout_logprobs"].tolist() == [[-1.0]]


def test_rollout_context_forward_batch_swaps_sequences_and_keeps_the_rest():
    batch = _logprob_phase_batch()
    batch["rollout_routed_experts"] = torch.zeros(2, 2, 1, 1, dtype=torch.int64)
    forward = batch.select(["sequences", "attention_mask", "rollout_routed_experts"], metadata_keys=["response_length"])
    swapped = rollout_context_forward_batch(batch, forward)
    assert swapped["sequences"].tolist() == batch[ROLLOUT_SEQUENCES_KEY].tolist()
    assert swapped["attention_mask"].tolist() == batch[ROLLOUT_ATTENTION_MASK_KEY].tolist()
    assert torch.equal(swapped["rollout_routed_experts"], forward["rollout_routed_experts"])
    assert swapped.metadata == {"response_length": 2}
    # the training-context forward batch is untouched
    assert forward["sequences"].tolist() == batch["sequences"].tolist()


def test_rollout_context_forward_batch_is_none_without_edited_rows():
    batch = _logprob_phase_batch(edited=(False, False))
    forward = batch.select(["sequences", "attention_mask"], metadata_keys=["response_length"])
    assert rollout_context_forward_batch(batch, forward) is None


# ---------------------------------------------------------------------------
# fwd_logprobs_values_reward: which forward sees which context
# ---------------------------------------------------------------------------
class _ActorGroup:
    """Records every forward's batch and answers with pre-set logprob tensors, in call order."""

    def __init__(self, outputs):
        self.actor_infos = [SimpleNamespace(rank=SimpleNamespace(dp_size=1))]
        self.forward_batches = []
        self._outputs = list(outputs)

    def async_run_ray_method(self, dispatch_type, method_name, data=None):
        del dispatch_type
        if method_name == "forward":
            self.forward_batches.append(data)
            output = TrainingOutputBatch({"output": self._outputs.pop(0)})
            output.metadata = dict(data.metadata)
            return [output]
        if method_name == "empty_cache":
            return []
        raise AssertionError(f"unexpected method {method_name}")

    def backload_to_gpu(self, *args, **kwargs):
        pass

    def offload_to_cpu(self, *args, **kwargs):
        pass


BASE = torch.tensor([[-1.1, -2.2], [-1.3, -0.7]])


def _phase_trainer(monkeypatch, *, colocate_all, **context_distillation):
    cfg = example_dummy_config()
    cfg.trainer.algorithm.context_distillation.enabled = True
    for key, value in context_distillation.items():
        cfg.trainer.algorithm.context_distillation[key] = value
    cfg.trainer.placement.colocate_policy_ref = colocate_all
    cfg.generator.sampling_params.logprobs = 0
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.colocate_all = colocate_all
    trainer.critic_model = None
    trainer.global_step = 3
    trainer.all_metrics = {}
    monkeypatch.setattr(trainer_module.ray, "get", lambda refs: refs)
    monkeypatch.setattr(trainer_module, "concatenate_outputs_after_mesh_dispatch", lambda infos, results: results[0])
    return trainer


@pytest.mark.parametrize("colocate_all", [False, True])
def test_logprob_phase_forwards_edited_rows_under_the_rollout_context(monkeypatch, colocate_all):
    trainer = _phase_trainer(monkeypatch, colocate_all=colocate_all)
    trainer.policy_model = _ActorGroup([TRAINING, ROLLOUT])
    trainer.ref_model = _ActorGroup([BASE])

    out = trainer.fwd_logprobs_values_reward(_logprob_phase_batch())

    # kl_reference=rollout: the frozen reference saw the served prompts
    assert trainer.ref_model.forward_batches[0]["sequences"].shape == (2, 5)
    # policy: training context first, then the rollout context for the TIS reference
    assert [batch["sequences"].shape for batch in trainer.policy_model.forward_batches] == [(2, 4), (2, 5)]
    torch.testing.assert_close(out["action_log_probs"], TRAINING)
    torch.testing.assert_close(out["base_action_log_probs"], BASE)
    torch.testing.assert_close(out["rollout_logprobs"], rebase_behavior_logprobs(ENGINE, TRAINING, ROLLOUT, EDITED))
    assert not any(key in out for key in ROLLOUT_CONTEXT_TENSOR_KEYS)
    assert trainer.all_metrics[EDITED_FRACTION_METRIC] == pytest.approx(0.5)
    assert "policy/rollout_train_prob_diff_mean" in trainer.all_metrics
    assert "reward/policy_ref_kl" in trainer.all_metrics


def test_logprob_phase_can_keep_the_self_reference_and_skip_the_second_forward(monkeypatch):
    trainer = _phase_trainer(monkeypatch, colocate_all=False, tis_reference="none", kl_reference="training")
    trainer.policy_model = _ActorGroup([TRAINING])
    trainer.ref_model = _ActorGroup([BASE])

    out = trainer.fwd_logprobs_values_reward(_logprob_phase_batch())

    assert trainer.ref_model.forward_batches[0]["sequences"].shape == (2, 4)
    assert [batch["sequences"].shape for batch in trainer.policy_model.forward_batches] == [(2, 4)]
    torch.testing.assert_close(out["rollout_logprobs"], neutralize_behavior_logprobs(ENGINE, TRAINING, EDITED))
    assert trainer.all_metrics[SHIFT_PER_TOKEN_METRIC] == 0.0


def test_logprob_phase_skips_the_extra_forwards_when_no_row_is_edited(monkeypatch):
    trainer = _phase_trainer(monkeypatch, colocate_all=False)
    trainer.policy_model = _ActorGroup([TRAINING])
    trainer.ref_model = _ActorGroup([BASE])

    out = trainer.fwd_logprobs_values_reward(_logprob_phase_batch(edited=(False, False)))

    assert trainer.ref_model.forward_batches[0]["sequences"].shape == (2, 4)
    assert len(trainer.policy_model.forward_batches) == 1
    torch.testing.assert_close(out["rollout_logprobs"], ENGINE)
    assert trainer.all_metrics[EDITED_FRACTION_METRIC] == 0.0
