"""The chunked logprob backward must stay bit-identical after dropping its one-hot temporaries."""

import pytest
import torch

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.distributed.megatron import model_utils  # noqa: E402
from skyrl_train.distributed.megatron.model_utils import (  # noqa: E402
    ChunkedDistributedLogprob,
    _compute_distributed_log_softmax,
)


def _reference_backward(vocab_parallel_logits, target_mask, masked_target, grad_output, chunk_size, group):
    """The one-hot formulation the module replaced, kept verbatim as the oracle."""
    partition_vocab_size = int(vocab_parallel_logits.shape[-1])
    seq_size = int(vocab_parallel_logits.shape[1])
    num_chunks = (seq_size + chunk_size - 1) // chunk_size
    grad_input = torch.empty_like(vocab_parallel_logits)
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(seq_size, (chunk_idx + 1) * chunk_size)
        logits = vocab_parallel_logits[:, chunk_start:chunk_end, :].to(dtype=torch.float32)
        softmax_output = _compute_distributed_log_softmax(logits, group=group).exp()
        is_chosen = (~(target_mask[:, chunk_start:chunk_end])).unsqueeze(-1) * torch.nn.functional.one_hot(
            masked_target[:, chunk_start:chunk_end],
            num_classes=partition_vocab_size,
        )
        grad_input_chunk = grad_input[:, chunk_start:chunk_end, :]
        grad_input_chunk.copy_(is_chosen.float().sub_(softmax_output))
        grad_input_chunk.mul_(grad_output[:, chunk_start:chunk_end].unsqueeze(dim=-1))
    return grad_input


def _bits(tensor):
    return tensor.contiguous().view(torch.int16 if tensor.dtype is torch.bfloat16 else torch.int32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [1, 3, 5])
@pytest.mark.parametrize("seed", range(3))
def test_chunked_backward_matches_one_hot_reference_bitwise(single_rank_group, dtype, chunk_size, seed):
    torch.manual_seed(seed)
    batch, seq, vocab = 2, 7, 12
    vocab_start, vocab_end = 3, 3 + vocab
    # Wide logit gaps underflow some softmax entries to zero, so the sign of zero is covered too.
    logits = (torch.randn(batch, seq, vocab) * 40).to(dtype)
    # Targets outside this partition exercise the masked rows.
    targets = torch.randint(0, vocab_end + 3, (batch, seq))
    grad_output = torch.randn(batch, seq)

    target_mask = (targets < vocab_start) | (targets >= vocab_end)
    masked_target = targets - vocab_start
    masked_target[target_mask] = 0
    expected = _reference_backward(logits, target_mask, masked_target, grad_output, chunk_size, single_rank_group)

    leaf = logits.clone().requires_grad_(True)
    log_probs = ChunkedDistributedLogprob.apply(
        leaf, targets, vocab_start, vocab_end, chunk_size, single_rank_group, False
    )
    log_probs.backward(grad_output)

    assert leaf.grad.dtype is dtype
    assert torch.equal(_bits(leaf.grad), _bits(expected))


def test_chunked_backward_builds_no_one_hot(single_rank_group, monkeypatch):
    def reject_one_hot(*args, **kwargs):
        raise AssertionError("chunked backward built a one-hot temporary")

    monkeypatch.setattr(model_utils.torch.nn.functional, "one_hot", reject_one_hot)
    logits = torch.randn(1, 6, 8, requires_grad=True)
    targets = torch.randint(0, 8, (1, 6))

    ChunkedDistributedLogprob.apply(logits, targets, 0, 8, 2, single_rank_group, False).sum().backward()

    assert logits.grad is not None
