"""vLLM's FA3 prefill call on synthetic steps, against the trainer's ``fa3_attention`` call.

Three properties the trainer flag relies on, checked on the kernel itself with random inputs:

- ``fa3_split_counts`` gives each request of a step the split count vLLM's call used: a call on that
  request alone, with the count as its cap, returns the step's rows byte for byte;
- the trainer's contiguous-key call (``fa3_attention_sbhd``) returns the same bytes as vLLM's paged cache;
- with one split, a request's rows do not depend on where chunked prefill cut it, while the key window
  does not reach past position 0 (compared against the whole sequence in one unsplit call).
"""

from __future__ import annotations

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

from skyrl_train.mismatch_harness.vllm_side import (
    FLASH_ATTENTION_VERSION,
    GrugShape,
    Requests,
    fa3_num_splits,
    flash_attention_prefill,
    paged_kv_cache,
)
from skyrl_train.models.grug_vllm_kernels import Fa3Request, fa3_attention_plan, fa3_attention_sbhd, fa3_split_counts

# Prefill steps (query tokens per request, each a whole sequence): single requests on both sides of the
# 512-token CUDA-graph cap and of FA3's occupancy heuristic, and mixed steps.
STEPS = (
    (300,),
    (299,),
    (600,),
    (700,),
    (100, 150, 200),
    (512,),
    (513,),
    (40, 460),
    (1200, 800),
    (2000,),
    (16, 16, 16),
    (1900, 60),
)
# (sequence length, chunk start): the second chunk of a prefill cut at an arbitrary token.
CHUNKS = ((2000, 700), (2000, 1037), (3000, 1037))


def _equal_fraction(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left.view(torch.int16) == right.view(torch.int16)).float().mean().item()


def _trainer_call(query, key, value, *, window, scale, splits: int | None) -> torch.Tensor:
    """The trainer flag's call on one sequence (``[tokens, heads, dim]``), with or without a split plan."""
    tokens, heads, head_dim = query.shape
    arguments = (query[:, None], key[:, None], value[:, None])
    if splits is None:
        return fa3_attention_sbhd(*arguments, window=window, scale=scale).view(tokens, heads, head_dim)
    with fa3_attention_plan([splits], [tokens]):
        return fa3_attention_sbhd(*arguments, window=window, scale=scale).view(tokens, heads, head_dim)


def _chunk_rows(query, key, value, start: int, *, window, scale) -> torch.Tensor:
    """vLLM's call for the rows ``[start, tokens)`` of one sequence, its earlier keys already in the cache."""
    tokens = key.shape[0]
    key_cache, value_cache, block_table = paged_kv_cache(key, value, (tokens,))
    rows = tokens - start
    output = torch.empty_like(query[start:])
    descale = torch.ones(1, key.shape[1], dtype=torch.float32, device=key.device)
    flash_attn_varlen_func(
        q=query[start:].contiguous(),
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32, device=key.device),
        max_seqlen_q=rows,
        seqused_k=torch.tensor([tokens], dtype=torch.int32, device=key.device),
        max_seqlen_k=tokens,
        softmax_scale=scale,
        causal=True,
        window_size=[window - 1, 0] if window is not None else None,
        block_table=block_table,
        softcap=0,
        scheduler_metadata=None,
        fa_version=FLASH_ATTENTION_VERSION,
        k_descale=descale,
        v_descale=descale,
        num_splits=fa3_num_splits(rows),
    )
    return output


@torch.no_grad()
def fa3_split_check(shape: GrugShape, generator: torch.Generator) -> dict[str, list[dict]]:
    """Run every step and chunk case on random bf16 inputs; returns one record per request and case."""
    scale = shape.head_dim**-0.5

    def random(tokens: int, heads: int) -> torch.Tensor:
        return torch.randn(tokens, heads, shape.head_dim, generator=generator, device="cuda").to(torch.bfloat16)

    steps = []
    for lengths in STEPS:
        for window in (None, shape.sliding_window):
            total = sum(lengths)
            query, key, value = random(total, shape.heads), random(total, shape.kv_heads), random(total, shape.kv_heads)
            step = flash_attention_prefill(
                query, key, value, Requests(lengths), window=window, scale=scale, num_splits=fa3_num_splits(total)
            )
            counts = fa3_split_counts(
                [Fa3Request(length, length) for length in lengths],
                kv_heads=shape.kv_heads,
                query_heads_per_kv_head=shape.heads // shape.kv_heads,
                window=window,
            )
            start = 0
            for length, splits in zip(lengths, counts, strict=True):
                rows = slice(start, start + length)
                alone = flash_attention_prefill(
                    query[rows],
                    key[rows],
                    value[rows],
                    Requests((length,)),
                    window=window,
                    scale=scale,
                    num_splits=splits,
                )
                planned = _trainer_call(query[rows], key[rows], value[rows], window=window, scale=scale, splits=splits)
                unsplit = _trainer_call(query[rows], key[rows], value[rows], window=window, scale=scale, splits=None)
                steps.append(
                    {
                        "step": list(lengths),
                        "window": window,
                        "tokens": length,
                        "splits": splits,
                        "paged_alone_equal_fraction": _equal_fraction(alone, step[rows]),
                        "trainer_planned_equal_fraction": _equal_fraction(planned, step[rows]),
                        "trainer_unsplit_equal_fraction": _equal_fraction(unsplit, step[rows]),
                    }
                )
                start += length
    chunks = []
    for tokens, start in CHUNKS:
        for window in (None, shape.sliding_window):
            query, key, value = (
                random(tokens, shape.heads),
                random(tokens, shape.kv_heads),
                random(tokens, shape.kv_heads),
            )
            chunk = _chunk_rows(query, key, value, start, window=window, scale=scale)
            whole = _trainer_call(query, key, value, window=window, scale=scale, splits=None)[start:]
            record = {
                "tokens": tokens,
                "chunk_start": start,
                "window": window,
                "chunk_splits": fa3_split_counts(
                    [Fa3Request(tokens - start, tokens)],
                    kv_heads=shape.kv_heads,
                    query_heads_per_kv_head=shape.heads // shape.kv_heads,
                    window=window,
                )[0],
                "equal_fraction": _equal_fraction(chunk, whole),
            }
            inside = (window or tokens) - start
            if 0 < inside < tokens - start:
                record["equal_fraction_before_window"] = _equal_fraction(chunk[:inside], whole[:inside])
                record["equal_fraction_past_window"] = _equal_fraction(chunk[inside:], whole[inside:])
            chunks.append(record)
    return {"steps": steps, "chunks": chunks}
