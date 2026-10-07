from types import SimpleNamespace

import torch

from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.objective.losses import PolicyLossInputs  # noqa: F401 — populate the loss registry
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import StepCounts, WeightCounts
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingBatchIterator
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from tests.cpu.util import dpo_test_config


def pair_trajectory_batch():
    return {
        "prompt_token_ids": [[1, 2], [1, 2]],
        "response_ids": [[4, 5, 6], [7]],
        "rewards": [[0.0, 0.0, 1.0], [0.0]],
        "loss_masks": [[1, 1, 1], [1]],
        "rollout_logprobs": None,
        "stop_reasons": ["preference_pair", "preference_pair"],
        "verification_results": [None, None],
        "evidence_messages": [None, None],
        "pair_roles": [1, -1],
    }


def test_pair_roles_survive_conversion_into_experience():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = dpo_test_config()
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.pad_batch = lambda batch: batch
    trainer.trajectory_selector = None
    trainer.global_step = 0
    trainer._training_metrics_enabled = False
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=2)

    batch = trainer.convert_to_training_input(pair_trajectory_batch(), ["a", "a"])
    assert torch.equal(batch["pair_roles"], torch.tensor([1.0, -1.0]))

    # The logprob forward pass fills these before the learner consumes the batch.
    batch["action_log_probs"] = torch.tensor([[-0.5, -0.6, -0.7], [-0.8, 0.0, 0.0]])
    batch["base_action_log_probs"] = torch.tensor([[-0.4, -0.5, -0.6], [-0.9, 0.0, 0.0]])
    batch["values"] = torch.zeros_like(batch["action_log_probs"])
    batch["returns"] = torch.zeros_like(batch["action_log_probs"])
    batch["advantages"] = torch.zeros_like(batch["action_log_probs"])

    [experience] = list(TrainingBatchIterator(batch, sample_batch_size=2))
    assert torch.equal(experience.pair_roles, torch.tensor([1.0, -1.0]))

    micro = build_objective_micro_batch(
        action_log_probs=experience.action_log_probs,
        old_action_log_probs=experience.action_log_probs,
        base_action_log_probs=torch.zeros_like(experience.action_log_probs),
        advantages=experience.advantages,
        loss_mask=experience.loss_mask,
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros_like(experience.action_log_probs),
        think_token_weight=1.0,
        teacher=None,
        pair_roles=experience.pair_roles,
    )
    assert micro.policy.dpo is not None
    assert micro.policy.ref_log_probs is not None

    counts = StepCounts(
        policy=WeightCounts(tokens=4.0, rows=2.0),
        mask=WeightCounts(tokens=4.0, rows=2.0),
        teacher=WeightCounts(tokens=0.0, rows=0.0),
        nonzero_advantage_rows=0.0,
        max_seq_len=8,
    )
    objective = compute_policy_objective(
        micro,
        loss=PolicyLossRegistry.get("dpo").function,
        counts=counts,
        config=trainer.cfg.trainer.algorithm,
        loss_scale=1.0,
        report_scale=1.0,
    )
    assert objective.metrics["dpo/loss"] > 0
    # The policy assigns the chosen completion less probability than the reference: accuracy 0.
    assert objective.metrics["dpo/accuracy"] == 0.0
    assert torch.isfinite(objective.optimization_loss)
