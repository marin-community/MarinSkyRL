"""Guards for the vocab-parallel logprob chunking path (logprob_chunk_size).

Why this exists: a 30B agentic RL run OOM'd at global_step 1 while computing
log-probs over a ~146k-token trajectory — the unchunked path materializes a single
[num_tokens, vocab//TP] fp32 logits tensor (~21 GiB for 146k tokens at TP4) inside
`DistributedLogprob`. The fix flips the base config default `logprob_chunk_size`
from null (unchunked) to 1024, which routes both the policy and ref logprob forwards
through `ChunkedDistributedLogprob` (per-position log-softmax, chunked along the
sequence dim), bounding peak memory regardless of sequence length.

These tests pin the two properties that make that default flip safe:
  1. The chunked path is numerically identical to the unchunked path — in both the
     forward log-probs AND the backward gradient — so turning it on cannot change
     training results. (Log-softmax is per-position over vocab; chunking only splits
     the independent sequence positions, so it is exact, not approximate.)
  2. The composed base config defaults `logprob_chunk_size` to 1024 for BOTH the
     policy and ref megatron_config (the two keys the workers read).

`model_utils` imports `megatron.core.parallel_state` at module load, but the functions
under test never touch it, so when megatron is not installed (the CPU CI env) we stub
that one submodule — only if megatron is genuinely absent, so a real-megatron env is
left untouched.
"""

import importlib.util
import os
import sys
import types

import pytest
import torch
import torch.distributed as dist

if importlib.util.find_spec("megatron") is None:
    for _name in ("megatron", "megatron.core", "megatron.core.parallel_state"):
        sys.modules.setdefault(_name, types.ModuleType(_name))

from skyrl_train.distributed.megatron.model_utils import from_parallel_logits_to_logprobs  # noqa: E402


@pytest.fixture(scope="module")
def single_rank_group():
    """A world-size-1 gloo process group so the TP all-reduces inside the logprob
    kernels are no-ops (vocab_start=0, vocab_end=full vocab) and the computation runs
    on a single CPU process."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    created = False
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
        created = True
    try:
        yield dist.group.WORLD
    finally:
        if created:
            dist.destroy_process_group()


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
