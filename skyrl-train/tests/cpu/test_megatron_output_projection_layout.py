import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from skyrl_train.workers.megatron.output_projection_layout import use_batch_major_output_projection


class ColumnProjection(nn.Linear):
    """Real linear projection with Megatron's logits/bias return contract."""

    def forward(self, inputs):
        return super().forward(inputs), None


@pytest.mark.parametrize("scale", [1.0, 0.25])
def test_batch_major_projection_reuses_logits_storage_and_preserves_dpo_gradients(scale):
    generator = torch.Generator().manual_seed(9)
    head = ColumnProjection(7, 13, bias=False, dtype=torch.float64)
    reference_head = copy.deepcopy(head)
    hidden = torch.randn(11, 2, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    targets = torch.randint(13, (2, 11), generator=generator)
    masks = torch.tensor([[0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]])
    use_batch_major_output_projection(head)

    sequence_logits, _ = head(hidden)
    sequence_logits = sequence_logits * scale
    actual_logits = sequence_logits.transpose(0, 1).contiguous()
    expected_logits = F.linear(reference_hidden, reference_head.weight).transpose(0, 1).contiguous() * scale
    assert actual_logits.untyped_storage().data_ptr() == sequence_logits.untyped_storage().data_ptr()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-12, atol=1e-12)

    def dpo_loss(logits):
        token_logprobs = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        sums = (token_logprobs * masks).sum(-1)
        return -F.logsigmoid(0.1 * (sums[0] - sums[1] - 0.75))

    actual_loss = dpo_loss(actual_logits)
    expected_loss = dpo_loss(expected_logits)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(head.weight.grad, reference_head.weight.grad, rtol=1e-12, atol=1e-12)
