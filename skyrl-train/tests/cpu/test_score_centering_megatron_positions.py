"""Selected behavior-head IDs must follow Megatron's left-padding compaction."""

import torch

from skyrl_train.distillation import student_topk_logprobs_for_compacted_response


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
