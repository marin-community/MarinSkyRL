"""Guards for the vocab-parallel logprob chunking path (logprob_chunk_size).

Why this exists: a 30B agentic RL run OOM'd at global_step 1 while computing
log-probs over a ~146k-token trajectory — the unchunked path materializes a single
[num_tokens, vocab//TP] fp32 logits tensor (~21 GiB for 146k tokens at TP4) inside
`DistributedLogprob`. The fix flips the base config default `logprob_chunk_size`
from null (unchunked) to 1024, which routes both the policy and ref logprob forwards
through `ChunkedDistributedLogprob` (per-position log-softmax, chunked along the
sequence dim), bounding peak memory regardless of sequence length.

These tests pin the properties that make the Megatron logprob and entropy paths safe:
  1. The chunked path is numerically identical to the unchunked path — in both the
     forward log-probs AND the backward gradient — so turning it on cannot change
     training results. (Log-softmax is per-position over vocab; chunking only splits
     the independent sequence positions, so it is exact, not approximate.)
  2. The composed base config defaults `logprob_chunk_size` to 1024 for BOTH the
     policy and ref megatron_config (the two keys the workers read).
  3. Entropy and logprob losses can backpropagate through the same logits tensor,
     and their combined gradient matches an independent PyTorch reference.

`model_utils` imports `megatron.core.parallel_state` at module load, but the functions
under test only need one tensor-parallel group lookup. When megatron is not installed
(the CPU CI env), the shared tests/cpu/util.py helper stubs the import and the entropy
test supplies the real single-rank process group. A real-megatron env is left untouched.
"""

import pytest
import torch

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.distributed.megatron import model_utils  # noqa: E402
from skyrl_train.distributed.megatron.model_utils import (  # noqa: E402
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
    vocab_parallel_entropy,
)


def _logprobs(logits, targets, group, chunk_size, inference_only):
    return from_parallel_logits_to_logprobs(
        logits,
        targets,
        vocab_start_index=0,
        vocab_end_index=logits.shape[-1],
        tp_group=group,
        inference_only=inference_only,
        cp_group=None,
        chunk_size=chunk_size,
    )


def _packed_batch(logits, tokens, attention_mask):
    """Pack valid token/logit rows and record each sequence's packed offset."""
    packed_logits = torch.cat([row[mask] for row, mask in zip(logits, attention_mask, strict=True)]).unsqueeze(0)
    packed_tokens = torch.cat([row[mask] for row, mask in zip(tokens, attention_mask, strict=True)]).unsqueeze(0)
    lengths = attention_mask.sum(dim=1, dtype=torch.int32)
    offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    return packed_logits, packed_tokens, offsets


# chunk sizes that exercise: divides the seq exactly, does NOT divide (ragged last
# chunk), == 1 (max chunking), and >= seq (single chunk == unchunked fast path).
@pytest.mark.parametrize("chunk_size", [1, 3, 4, 7, 16, 100])
def test_chunked_matches_unchunked_forward(single_rank_group, chunk_size):
    torch.manual_seed(0)
    batch, seq, vocab = 2, 13, 32
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32)
    targets = torch.randint(0, vocab, (batch, seq))

    unchunked = _logprobs(logits, targets, single_rank_group, None, inference_only=True)
    chunked = _logprobs(logits, targets, single_rank_group, chunk_size, inference_only=True)

    assert chunked.shape == unchunked.shape
    # per-position log-softmax => exact, not approximate
    assert torch.allclose(chunked, unchunked, atol=1e-6, rtol=1e-6), (
        f"chunk_size={chunk_size} max|diff|={(chunked - unchunked).abs().max().item()}"
    )


@pytest.mark.parametrize("chunk_size", [1, 4, 7])
def test_chunked_matches_unchunked_backward(single_rank_group, chunk_size):
    torch.manual_seed(1)
    batch, seq, vocab = 2, 11, 24
    base = torch.randn(batch, seq, vocab, dtype=torch.float32)
    targets = torch.randint(0, vocab, (batch, seq))

    def grad_for(cs):
        logits = base.clone().requires_grad_(True)
        out = _logprobs(logits, targets, single_rank_group, cs, inference_only=False)
        out.sum().backward()
        return logits.grad

    g_unchunked = grad_for(None)
    g_chunked = grad_for(chunk_size)

    assert torch.allclose(g_chunked, g_unchunked, atol=1e-6, rtol=1e-6), (
        f"chunk_size={chunk_size} grad max|diff|={(g_chunked - g_unchunked).abs().max().item()}"
    )


def test_chunked_bf16_model_output_matches_lossless_fp32_output(single_rank_group):
    """Keeping MCore logits in bf16 must preserve the fp32 logprob result and model gradient.

    MCore's ``Float16Module`` otherwise expands the entire last-pipeline-stage vocab
    tensor to fp32. The chunked logprob path casts each chunk itself, so it must agree
    with the old lossless bf16-to-fp32 output conversion without allocating the full
    fp32 tensor first.
    """
    torch.manual_seed(3)
    batch, seq, vocab = 2, 11, 32
    model_precision_logits = torch.randn(batch, seq, vocab, dtype=torch.bfloat16)
    targets = torch.randint(0, vocab, (batch, seq))

    direct_bf16 = model_precision_logits.clone().requires_grad_(True)
    old_fp32_input = model_precision_logits.clone().requires_grad_(True)

    direct_logprobs = _logprobs(direct_bf16, targets, single_rank_group, chunk_size=3, inference_only=False)
    old_logprobs = _logprobs(old_fp32_input.float(), targets, single_rank_group, chunk_size=3, inference_only=False)

    assert torch.equal(direct_logprobs, old_logprobs)

    direct_logprobs.sum().backward()
    old_logprobs.sum().backward()
    # The custom backward must return the model activation dtype. Returning a
    # full fp32 [B, S, V_local] buffer is the late-step RL OOM this test guards.
    assert direct_bf16.grad.dtype is torch.bfloat16
    assert torch.equal(direct_bf16.grad, old_fp32_input.grad)


def test_chunked_backward_does_not_allocate_a_full_fp32_gradient(single_rank_group, monkeypatch):
    """Chunked backward may only allocate full-vocab storage in model dtype.

    ``torch.zeros_like(..., dtype=float32)`` was an overlooked second full
    vocab-parallel reconstruction: its 12.9 GiB allocation killed the r6
    Pymethods and r9 OpenCode arms after several healthy steps.  The output
    gradient is filled chunk-by-chunk, so a zeroed fp32 destination is neither
    necessary nor safe at this sequence length.
    """
    torch.manual_seed(4)
    logits = torch.randn(1, 9, 16, dtype=torch.bfloat16).requires_grad_(True)
    targets = torch.randint(0, 16, (1, 9))

    def reject_full_fp32_zeros_like(*args, **kwargs):
        if kwargs.get("dtype") is torch.float32:
            raise AssertionError("chunked backward allocated a full fp32 gradient")
        return original_zeros_like(*args, **kwargs)

    original_zeros_like = model_utils.torch.zeros_like
    monkeypatch.setattr(model_utils.torch, "zeros_like", reject_full_fp32_zeros_like)

    _logprobs(logits, targets, single_rank_group, chunk_size=3, inference_only=False).sum().backward()
    assert logits.grad is not None
    assert logits.grad.dtype is torch.bfloat16


def test_vocab_parallel_entropy_and_logprob_share_logits_without_corrupting_backward(single_rank_group, monkeypatch):
    """Entropy and policy losses must backpropagate through the same logits tensor."""
    monkeypatch.setattr(model_utils.mpu, "get_tensor_model_parallel_group", lambda: single_rank_group, raising=False)
    torch.manual_seed(5)
    base_logits = torch.randn(2, 7, 16, dtype=torch.float32)
    targets = torch.randint(0, base_logits.shape[-1], (base_logits.shape[0], base_logits.shape[1]))

    parallel_logits = base_logits.clone().requires_grad_(True)
    parallel_logprobs = _logprobs(
        parallel_logits,
        targets,
        single_rank_group,
        chunk_size=3,
        inference_only=False,
    )
    parallel_entropy = vocab_parallel_entropy(parallel_logits)
    (parallel_logprobs.sum() + 0.003 * parallel_entropy.sum()).backward()

    reference_logits = base_logits.clone().requires_grad_(True)
    reference_log_probs = reference_logits.log_softmax(dim=-1)
    rolled_targets = targets.roll(shifts=-1, dims=-1)
    reference_chosen = reference_log_probs.gather(-1, rolled_targets.unsqueeze(-1)).squeeze(-1)[:, :-1]
    reference_entropy = -(reference_log_probs.exp() * reference_log_probs).sum(dim=-1)
    (reference_chosen.sum() + 0.003 * reference_entropy.sum()).backward()

    assert torch.allclose(parallel_logits.grad, reference_logits.grad, atol=1e-6, rtol=1e-6)


def test_packed_logprobs_preserve_left_padded_action_positions(single_rank_group):
    """Packed logprobs must land at the original action positions, not at column zero."""
    torch.manual_seed(2)
    batch, seq, vocab, num_actions = 2, 9, 32, 3
    base_logits = torch.randn(batch, seq, vocab, dtype=torch.float32)
    padded_logits = base_logits.clone().requires_grad_(True)
    packed_source_logits = base_logits.clone().requires_grad_(True)
    tokens = torch.randint(0, vocab, (batch, seq))
    attention_mask = torch.tensor(
        [
            [False, False, True, True, True, True, True, True, True],
            [False, False, False, True, True, True, True, True, True],
        ]
    )
    packed_logits, packed_tokens, offsets = _packed_batch(packed_source_logits, tokens, attention_mask)

    padded = _logprobs(padded_logits, tokens, single_rank_group, chunk_size=3, inference_only=False)
    packed = from_parallel_logits_to_logprobs_packed_sequences(
        packed_logits,
        packed_tokens,
        offsets,
        attention_mask=attention_mask,
        vocab_start_index=0,
        vocab_end_index=vocab,
        group=single_rank_group,
        inference_only=False,
        cp_group=None,
        chunk_size=3,
    )

    assert torch.allclose(packed[:, -num_actions:], padded[:, -num_actions:], atol=1e-6, rtol=1e-6)

    padded[:, -num_actions:].sum().backward()
    packed[:, -num_actions:].sum().backward()
    assert torch.allclose(
        packed_source_logits.grad[attention_mask], padded_logits.grad[attention_mask], atol=1e-6, rtol=1e-6
    )


def test_base_config_defaults_chunk_size_for_policy_and_ref():
    """The composed base config must default logprob_chunk_size to a non-null int for
    BOTH the policy and ref megatron_config — the two keys the workers read. Guards
    against a silent regression back to the unchunked (OOM-prone) default."""
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf
    from skyrl_train.config.utils import get_default_config

    cfg = get_default_config()
    policy = OmegaConf.select(cfg, "trainer.policy.megatron_config.logprob_chunk_size")
    ref = OmegaConf.select(cfg, "trainer.ref.megatron_config.logprob_chunk_size")

    assert policy == 1024, f"policy logprob_chunk_size default = {policy!r}, expected 1024"
    assert ref == 1024, f"ref logprob_chunk_size default = {ref!r}, expected 1024"
