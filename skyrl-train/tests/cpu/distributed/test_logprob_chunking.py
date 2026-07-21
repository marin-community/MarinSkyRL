"""Guards for the vocab-parallel logprob chunking path (logprob_chunk_size).

Why this exists: a 30B agentic RL run OOM'd at global_step 1 while computing
log-probs over a ~146k-token trajectory — the unchunked path materializes a single
[num_tokens, vocab//TP] fp32 logits tensor (~21 GiB for 146k tokens at TP4) inside
`DistributedLogprob`. The fix flips the base config default `logprob_chunk_size`
from null (unchunked) to an int, which routes both the policy and ref logprob
forwards through `ChunkedDistributedLogprob` (per-position log-softmax, chunked along
the sequence dim), bounding peak memory regardless of sequence length.

These tests pin the two properties that make that default flip safe:
  1. The chunked path is numerically identical to the unchunked path — in both the
     forward log-probs AND the backward gradient — so turning it on cannot change
     training results. (Log-softmax is per-position over vocab; chunking only splits
     the independent sequence positions, so it is exact, not approximate.)
  2. The MegatronModelWrapper honors an explicitly-passed chunk size and, when the
     caller omits it, falls back to the policy config key (back-compat). This is the
     plumbing that lets the policy worker read its own key and the ref worker read
     the ref key.
"""

import os
import types

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from skyrl_train.distributed.megatron.model_utils import from_parallel_logits_to_logprobs


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


def _make_wrapper(monkeypatch, cfg):
    """Construct a MegatronModelWrapper without any real Megatron module: stub out
    get_model_config so __init__ only exercises the config resolution we care about."""
    import skyrl_train.workers.megatron.megatron_model_wrapper as mw

    monkeypatch.setattr(mw, "get_model_config", lambda module: types.SimpleNamespace())
    return mw.MegatronModelWrapper(config=cfg, actor_module=[object()])


def _make_wrapper_explicit(monkeypatch, cfg, chunk_size):
    import skyrl_train.workers.megatron.megatron_model_wrapper as mw

    monkeypatch.setattr(mw, "get_model_config", lambda module: types.SimpleNamespace())
    return mw.MegatronModelWrapper(config=cfg, actor_module=[object()], logprob_chunk_size=chunk_size)


def _cfg(policy_chunk, ref_chunk):
    return OmegaConf.create(
        {
            "trainer": {
                "use_sample_packing": True,
                "policy": {"megatron_config": {"logprob_chunk_size": policy_chunk}},
                "ref": {"megatron_config": {"logprob_chunk_size": ref_chunk}},
            }
        }
    )


def test_wrapper_uses_explicit_chunk_size(monkeypatch):
    # Policy passes its key, ref passes its (different) key: each must be honored.
    cfg = _cfg(policy_chunk=1024, ref_chunk=512)
    assert _make_wrapper_explicit(monkeypatch, cfg, 1024)._logprob_chunk_size == 1024
    assert _make_wrapper_explicit(monkeypatch, cfg, 512)._logprob_chunk_size == 512


def test_wrapper_explicit_none_disables_chunking(monkeypatch):
    # An explicit None must NOT fall back to the config key — it means "disabled".
    cfg = _cfg(policy_chunk=1024, ref_chunk=1024)
    assert _make_wrapper_explicit(monkeypatch, cfg, None)._logprob_chunk_size is None


def test_wrapper_falls_back_to_policy_key(monkeypatch):
    # Back-compat: a caller that omits the arg reads trainer.policy.megatron_config.
    cfg = _cfg(policy_chunk=2048, ref_chunk=512)
    assert _make_wrapper(monkeypatch, cfg)._logprob_chunk_size == 2048
