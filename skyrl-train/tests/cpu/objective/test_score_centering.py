"""Behavior evidence must survive collation, serialization and microbatching."""

import numpy as np
import pytest
import torch

from skyrl_train.objective.score_centering import collate_score_centering, score_centering_diagnostics
from skyrl_train.training_batch import TrainingBatchIterator, TrainingInputBatch


def test_score_evidence_survives_collation_serialization_and_microbatching(tmp_path):
    trajectory = {
        "loss_masks": [[1, 1], [1]],
        "student_topk_indices": [np.array([[3, 4], [5, 6]]), np.array([[7, 8]])],
        "behavior_topk_logprobs": [np.array([[-0.8, -1.8], [-0.9, -1.9]]), np.array([[-1.0, -2.0]])],
    }
    mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
    ids, behavior = collate_score_centering(
        trajectory, [[3, 5], [7]], mask, 2, sampled_logprobs=torch.tensor([[-0.8, -0.9], [-1.0, 0]])
    )
    torch.testing.assert_close(ids[mask.bool()], torch.from_numpy(np.vstack(trajectory["student_topk_indices"])))
    torch.testing.assert_close(
        behavior[mask.bool()],
        torch.from_numpy(np.vstack(trajectory["behavior_topk_logprobs"])).float(),
        atol=0,
        rtol=0,
    )
    assert (ids[~mask.bool()] == -1).all()
    assert torch.isnan(behavior[~mask.bool()]).all()
    old = behavior + 0.3
    payload = {
        "sequences": torch.tensor([[1, 3, 5], [2, 7, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "response_mask": mask.bool(),
        "loss_mask": mask,
        "score_topk_indices": ids,
        "score_old_logprobs": old,
        "score_behavior_logprobs": behavior,
    }
    payload.update(
        {
            key: torch.zeros_like(mask)
            for key in ("action_log_probs", "base_action_log_probs", "values", "returns", "advantages")
        }
    )
    batch = TrainingInputBatch(payload)
    batch.metadata = {"response_length": 2}
    path = tmp_path / "batch.pt"
    torch.save(batch, path)
    restored = torch.load(path, weights_only=False)
    experiences = list(TrainingBatchIterator(restored, sample_batch_size=1))
    assert len(experiences) == 2
    for row, experience in enumerate(experiences):
        for name, expected in (
            ("score_topk_indices", ids),
            ("score_old_logprobs", old),
            ("score_behavior_logprobs", behavior),
        ):
            torch.testing.assert_close(
                getattr(experience, name), expected[row : row + 1], atol=0, rtol=0, equal_nan=True
            )


def test_sampled_loss_diagnostics_exclude_masked_tokens_and_bound_the_sample():
    head = torch.tensor([[[0.3, 0.5], [torch.nan, torch.nan], [0.1, 0.4], [0.4, 0.5]]]).log()
    current = torch.tensor([[-0.5, torch.nan, -1.0, -9.0]], requires_grad=True)
    behavior = torch.tensor([[-0.7, torch.nan, -0.6, -3.0]])
    correction = torch.tensor([[-2.0, torch.nan, 1.0, 100.0]], requires_grad=True)
    metrics = score_centering_diagnostics(
        head, head, head, current, behavior, correction, torch.tensor([[1, 0, 1, 1]]), 2
    )
    assert metrics["score_centering/sampled_tokens"] == 2
    assert metrics["score_centering/sampled_loss_mean"] == pytest.approx(-0.5)
    assert metrics["score_centering/sampled_loss_abs_mean"] == pytest.approx(1.5)
    assert metrics["score_centering/sampled_current_retained_mass"] == pytest.approx(0.65)
    assert metrics["score_centering/sampled_behavior_omitted_mass"] == pytest.approx(0.35)
    assert metrics["score_centering/sampled_current_behavior_log_ratio_mean"] == pytest.approx(-0.1)
    assert metrics["score_centering/sampled_current_behavior_log_ratio_abs_mean"] == pytest.approx(0.3)
