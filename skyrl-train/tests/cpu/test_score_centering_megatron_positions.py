"""Selected behavior-head IDs must follow Megatron's left-padding compaction."""

import pytest
import torch

from skyrl_train.distillation import (
    student_topk_logprobs_for_compacted_response,
    student_topk_logprobs_from_sampled_action_logprobs,
)


def test_selected_response_logprobs_follow_each_compacted_row():
    # Prompts are left padded, responses are right padded. The second row has
    # fewer real tokens, so its first response logit is at compact position 1.
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]])
    selected_ids = torch.tensor(
        [
            [[1, 3], [2, 4], [0, 5]],
            [[6, 1], [-1, -1], [-1, -1]],
        ]
    )
    logits = torch.randn((2, 6, 7), dtype=torch.float32, requires_grad=True)

    selected = student_topk_logprobs_for_compacted_response(logits, selected_ids, attention_mask)
    normalized = logits.log_softmax(dim=-1)
    expected = torch.stack(
        [
            normalized[0, 2, 1],
            normalized[0, 2, 3],
            normalized[0, 3, 2],
            normalized[0, 3, 4],
            normalized[0, 4, 0],
            normalized[0, 4, 5],
            normalized[1, 1, 6],
            normalized[1, 1, 1],
        ]
    )
    valid = selected_ids >= 0
    torch.testing.assert_close(selected[valid], expected, atol=1e-6, rtol=1e-6)
    assert torch.isnan(selected[~valid]).all()

    selected_gradient = torch.autograd.grad(selected[valid].sum(), logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(selected_gradient, expected_gradient, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_response_reuses_sampled_normalizer_without_full_float_logits(dtype):
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]])
    selected_ids = torch.tensor([[[1, 3], [2, 4], [0, 5]], [[6, 1], [-1, -1], [-1, -1]]])
    sampled_ids = torch.tensor([[3, 4, 5], [2, 0, 0]])
    logits = torch.randn((2, 6, 7), dtype=dtype, requires_grad=True)
    normalized = logits.float().log_softmax(dim=-1)
    compact_positions = torch.tensor([[2, 3, 4], [1, 0, 0]])
    batch_positions = torch.arange(2)[:, None]
    sampled_logprobs = normalized[batch_positions, compact_positions, sampled_ids]

    saved_tensors = []

    def save(tensor):
        saved_tensors.append((tensor.shape, tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        selected = student_topk_logprobs_from_sampled_action_logprobs(
            logits, selected_ids, sampled_ids, sampled_logprobs, attention_mask
        )

    valid = selected_ids >= 0
    expected = normalized[torch.arange(2)[:, None, None], compact_positions[:, :, None], selected_ids.clamp_min(0)]
    torch.testing.assert_close(selected[valid], expected[valid], atol=1e-5, rtol=1e-5)
    assert torch.isnan(selected[~valid]).all()
    if dtype == torch.bfloat16:
        assert (logits.shape, torch.float32) not in saved_tensors

    selected_gradient = torch.autograd.grad(selected[valid].sum(), logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected[valid].sum(), logits)[0]
    tolerance = 5e-3 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(selected_gradient, expected_gradient, atol=tolerance, rtol=tolerance)
