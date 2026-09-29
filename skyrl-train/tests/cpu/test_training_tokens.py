"""Budgeted runs train on an exact number of unpadded loss tokens."""

from dataclasses import asdict

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.training_tokens import TrainingTokens, loss_token_budget
from skyrl_train.tensor_math import masked_mean


def test_final_budget_masks_surplus_without_truncating_responses_and_survives_resume():
    batch = TrainingInputBatch(
        {
            "loss_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 1], [0, 0, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]),
        }
    )
    batch.metadata = {"pad_size": 1}
    state = TrainingTokens(loss=10)
    state.limit_loss(batch, 15)
    assert batch["loss_mask"].tolist() == [[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]]
    assert batch["response_mask"].sum().item() == 11
    values = torch.ones((3, 4), requires_grad=True)
    loss = sum(masked_mean(row, mask) for row, mask in zip(values, batch["loss_mask"], strict=True))
    loss.backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[1, 1:].count_nonzero() == 0
    assert values.grad[2].count_nonzero() == 0
    metrics = state.consume(batch)
    assert metrics["consumed/loss_total"] == 15
    assert metrics["consumed/response_total"] == 7
    assert metrics["consumed/input_total"] == 11
    assert metrics["consumed/sequences_total"] == 2
    restored = TrainingTokens(**asdict(state))
    with pytest.raises(ValueError, match="exhausted"):
        restored.limit_loss(batch, 15)


def test_budget_rejects_unaccounted_repeated_update_epochs():
    with pytest.raises(ValueError, match="one update epoch"):
        loss_token_budget(OmegaConf.create({"loss_token_budget": 100, "update_epochs_per_batch": 2}))
