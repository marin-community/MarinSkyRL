"""Response positions with real gradient support for the native correction fixture."""

import math

import torch

from skyrl_train.training_batch import TrainingInputBatch
from tests.offpolicy_mask_reference import minimal_m2_reference


def left_pad_correction_batch(batch: TrainingInputBatch) -> TrainingInputBatch:
    """Move existing padding left without changing any nonpadding token or its order."""
    sequences = batch["sequences"].clone()
    attention = batch["attention_mask"].clone()
    actions = batch.metadata["response_length"]
    for row in range(len(sequences)):
        valid = attention[row].bool()
        tokens = sequences[row, valid].clone()
        assert len(tokens) >= actions + 1
        padding = sequences[row, ~valid].clone()
        sequences[row] = torch.cat((padding, tokens))
        attention[row].zero_()
        attention[row, -len(tokens) :] = 1
    batch["sequences"] = sequences
    batch["attention_mask"] = attention
    # The last action is predicted at the penultimate token position.
    assert attention[:, -(actions + 1) :].bool().all()
    return batch


def m2_fixture_diagnostics(current, old, attention_mask):
    """Report realized float32 deltas and retained differentiable positions per row."""
    diagnostics = []
    for row in range(len(current)):
        delta = current[row : row + 1] - old[row : row + 1]
        selected = torch.ones_like(delta, dtype=torch.bool)
        removed, satisfied = minimal_m2_reference(delta, -torch.ones_like(delta), selected, 0.04)
        support = attention_mask[row].bool().clone()
        support[support.nonzero().flatten()[-1]] = False
        support = support[:-1][-current.shape[1] :].unsqueeze(0)
        differentiable = delta > math.log(0.8)
        retained = support & ~removed & differentiable
        diagnostics.append(
            {
                "row": row,
                "logprob_dtype": str(current.dtype),
                "logprob_min": float(current[row].min()),
                "logprob_max": float(current[row].max()),
                "realized_delta": delta.detach().cpu().tolist()[0],
                "lower_clip_log_bound": math.log(0.8),
                "selected_action_positions": int(selected.sum()),
                "supported_action_positions": int(support.sum()),
                "removed_positions": int(removed.sum()),
                "retained_gradient_positions": int(retained.sum()),
                "budget_satisfied": satisfied,
            }
        )
    return diagnostics
