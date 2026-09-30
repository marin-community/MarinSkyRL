"""How vLLM's kernels change the same rows' bytes with the size of the step, at Snowball shapes (one H100).

Measures, on random inputs, the fraction of output elements of a fixed set of rows that stay byte-equal
when the kernel runs them inside steps of other sizes:

- the dense GEMMs compiled vLLM issues as ``extern_kernels.mm`` (bf16, cuBLAS) and the fp32 router GEMM,
  at step sizes from a single decode row to a full 8,192-token prefill step, against the 304-row step
  the harness replays;
- the routed experts (``TritonExperts``; its tile config follows the gathered token count), per slot;
- FA3 for the last row of a sequence as a decode step computes it (one query row, the lone request's
  split count) against the same row in the unsplit prefill.

Example::

    python -m skyrl_train.mismatch_harness.batch_dependence --model s3://.../hf-bf16-vllm --output s3://.../bd/<name>
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from marinskyrl.resource_locator import join_resource_path
from vllm.vllm_flash_attn import flash_attn_varlen_func

from skyrl_train.io import io
from skyrl_train.mismatch_harness.expert_parallel import ReduceOrder
from skyrl_train.mismatch_harness.run import stage_config
from skyrl_train.mismatch_harness.vllm_side import (
    GrugShape,
    Requests,
    VllmMoe,
    fa3_num_splits,
    flash_attention_prefill,
    paged_kv_cache,
    vllm_config_context,
)
from skyrl_train.models.grug_vllm_kernels import Fa3Request, fa3_split_counts

REFERENCE_ROWS = 304
STEP_ROWS = (1, 8, 64, 256, 304, 512, 1024, 4096, 8192)
# Gathered MoE token counts: eight ranks of one decode row, of the harness's padded 304-row step, and of
# prefill steps of 1,024 and 8,192 tokens.
MOE_TOKENS = (8, 2432, 8192, 65536)
DECODE_LENGTHS = (300, 600, 900, 2000)


def equal_fraction(left: torch.Tensor, right: torch.Tensor) -> float:
    width = torch.int16 if left.dtype == torch.bfloat16 else torch.int32
    return (left.contiguous().view(width) == right.contiguous().view(width)).float().mean().item()


def gemm_rows(shape: GrugShape, generator: torch.Generator) -> dict[str, dict[str, float]]:
    """Each dense GEMM's first rows at every step size against the 304-row step."""
    hidden = shape.hidden
    projections = {
        "q_proj": (shape.heads * shape.head_dim, hidden, torch.bfloat16),
        "k_proj": (shape.kv_heads * shape.head_dim, hidden, torch.bfloat16),
        "o_proj": (hidden, shape.heads * shape.head_dim, torch.bfloat16),
        "attn_gate (24 padded)": (24, hidden, torch.bfloat16),
        "gated_norm_down": (128, hidden, torch.bfloat16),
        "gated_norm_up": (hidden, 128, torch.bfloat16),
        # Snowball's shared expert is 2,560 wide, the hidden size.
        "shared_gate": (hidden, hidden, torch.bfloat16),
        "shared_down": (hidden, hidden, torch.bfloat16),
        "router (fp32)": (shape.experts, hidden, torch.float32),
    }
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {}
    try:
        for name, (rows_out, depth, dtype) in projections.items():
            inputs = torch.randn(max(STEP_ROWS), depth, generator=generator, device="cuda").to(dtype)
            weight = torch.randn(rows_out, depth, generator=generator, device="cuda").to(dtype) / depth**0.5
            reference = torch.mm(inputs[:REFERENCE_ROWS], weight.t())
            results[name] = {
                str(rows): equal_fraction(
                    torch.mm(inputs[:rows], weight.t())[: min(rows, REFERENCE_ROWS)],
                    reference[: min(rows, REFERENCE_ROWS)],
                )
                for rows in STEP_ROWS
            }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return results


def moe_slots(shape: GrugShape, generator: torch.Generator) -> dict[str, float]:
    """Per-slot routed expert outputs of eight tokens with the Triton config of each gathered token count."""
    experts, intermediate, hidden = shape.experts, shape.intermediate, shape.hidden
    w13 = (torch.randn(experts, 2 * intermediate, hidden, generator=generator, device="cuda") / hidden**0.5).to(
        torch.bfloat16
    )
    w2 = (torch.randn(experts, hidden, intermediate, generator=generator, device="cuda") / intermediate**0.5).to(
        torch.bfloat16
    )
    tokens = torch.randn(8, hidden, generator=generator, device="cuda").to(torch.bfloat16)
    ids = torch.stack([torch.randperm(experts, generator=generator, device="cuda")[: shape.top_k] for _ in range(8)])
    weights = torch.rand(8, shape.top_k, generator=generator, device="cuda") * 2.5 / shape.top_k
    slots = {}
    for count in MOE_TOKENS:
        moe = VllmMoe(
            bias=torch.zeros(experts, device="cuda"),
            w13=w13,
            w2=w2,
            top_k=shape.top_k,
            ep_size=1,
            home_rank=0,
            order=ReduceOrder.RING,
            config_tokens=count,
        )
        _, slots[count] = moe.experts(tokens, weights, ids.to(torch.int32))
    reference = slots[MOE_TOKENS[1]]
    return {str(count): equal_fraction(value, reference) for count, value in slots.items()}


def decode_attention(shape: GrugShape, generator: torch.Generator) -> list[dict]:
    """The last row of a sequence from a lone decode step against the same row from the unsplit prefill."""
    scale = shape.head_dim**-0.5
    rows = []
    for length in DECODE_LENGTHS:
        for window in (None, shape.sliding_window):
            query = torch.randn(length, shape.heads, shape.head_dim, generator=generator, device="cuda").to(
                torch.bfloat16
            )
            key, value = (
                torch.randn(length, shape.kv_heads, shape.head_dim, generator=generator, device="cuda").to(
                    torch.bfloat16
                )
                for _ in range(2)
            )
            prefill = flash_attention_prefill(
                query, key, value, Requests((length,)), window=window, scale=scale, num_splits=1
            )
            key_cache, value_cache, block_table = paged_kv_cache(key, value, (length,))
            decoded = torch.empty_like(query[-1:])
            descale = torch.ones(1, shape.kv_heads, dtype=torch.float32, device="cuda")
            flash_attn_varlen_func(
                q=query[-1:].contiguous(),
                k=key_cache,
                v=value_cache,
                out=decoded,
                cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
                max_seqlen_q=1,
                seqused_k=torch.tensor([length], dtype=torch.int32, device="cuda"),
                max_seqlen_k=length,
                softmax_scale=scale,
                causal=True,
                window_size=[window - 1, 0] if window is not None else None,
                block_table=block_table,
                softcap=0,
                scheduler_metadata=None,
                fa_version=3,
                k_descale=descale,
                v_descale=descale,
                num_splits=fa3_num_splits(1),
            )
            splits = fa3_split_counts(
                [Fa3Request(1, length)],
                kv_heads=shape.kv_heads,
                query_heads_per_kv_head=shape.heads // shape.kv_heads,
                window=window,
            )[0]
            rows.append(
                {
                    "length": length,
                    "window": window,
                    "decode_splits": splits,
                    "equal_fraction": equal_fraction(decoded, prefill[-1:]),
                }
            )
    return rows


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory, vllm_config_context():
        shape = GrugShape.from_config(stage_config(args.model, Path(directory)))
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        results = {
            "gpu": torch.cuda.get_device_name(),
            "reference_rows": REFERENCE_ROWS,
            "gemm_rows": gemm_rows(shape, generator),
            "moe_slots": moe_slots(shape, generator),
            "decode_attention": decode_attention(shape, generator),
        }
    payload = json.dumps(results, indent=1, sort_keys=True)
    io.write_bytes_atomic(join_resource_path(args.output, "batch_dependence.json"), payload.encode())
    print(payload, flush=True)
    print("PASS mismatch harness batch dependence", args.output, flush=True)


if __name__ == "__main__":
    main()
