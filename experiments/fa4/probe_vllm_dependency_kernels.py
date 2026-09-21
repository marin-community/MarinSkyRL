"""Exercise the vLLM kernels affected by the TileLang/TokenSpeed pin update."""

from __future__ import annotations

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

import torch


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


def _sinkhorn(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = residual_flat @ fn.T
    mixes *= (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    expanded_scale = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * expanded_scale + hc_base
    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
    res_mix = _sinkhorn(res_mix, sinkhorn_repeat, hc_sinkhorn_eps)
    layer_input = (residual * pre_mix).sum(-2).bfloat16()
    return post_mix, res_mix, layer_input


def probe_tilelang_mhc() -> dict[str, object]:
    import vllm.model_executor.kernels.mhc  # noqa: F401

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, hc_mult, hidden_size = 4, 4, 4096
    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
    )
    num_mixes = 2 * hc_mult + hc_mult * hc_mult
    fn = (
        torch.randn(num_mixes, hc_mult * hidden_size, device=device) * 1.0e-4
    )
    hc_scale = torch.randn(3, device=device) * 0.1
    hc_base = torch.randn(num_mixes, device=device) * 0.1
    params = (1.0e-6, 1.0e-6, 1.0e-6, 1.0, 20)

    expected = _mhc_pre_reference(
        residual, fn, hc_scale, hc_base, *params
    )
    actual = torch.ops.vllm.mhc_pre_tilelang(
        residual, fn, hc_scale, hc_base, *params
    )
    torch.cuda.synchronize()

    names = ("post_mix", "res_mix", "layer_input")
    errors = {name: _error(got, want) for name, got, want in zip(names, actual, expected)}
    for name, got, want in zip(names, actual, expected):
        torch.testing.assert_close(got, want, atol=5.0e-2, rtol=1.0e-2, msg=name)
    return {"shape": list(residual.shape), "errors": errors}


def _attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    # Inputs are token-major. Expand the single KV head across query heads.
    q = query.float().transpose(0, 1)
    k = key.float().expand(-1, query.shape[1], -1).transpose(0, 1)
    v = value.float().expand(-1, query.shape[1], -1).transpose(0, 1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        q_len, kv_len = query.shape[0], key.shape[0]
        mask = torch.ones(q_len, kv_len, device=query.device, dtype=torch.bool).tril(
            diagonal=kv_len - q_len
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(scores.softmax(-1), v).transpose(0, 1)


def probe_tokenspeed_mla() -> dict[str, object]:
    from tokenspeed_mla import (
        get_num_sm,
        tokenspeed_mla_decode,
        tokenspeed_mla_prefill,
    )

    torch.manual_seed(1)
    device = torch.device("cuda")
    num_heads, kv_heads = 8, 1
    d_qk, d_v, seq_len = 192, 128, 64
    scale = 1.0 / math.sqrt(d_qk)
    query = torch.randn(
        seq_len, num_heads, d_qk, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(seq_len, kv_heads, d_qk, device=device, dtype=torch.bfloat16)
    value = torch.randn(seq_len, kv_heads, d_v, device=device, dtype=torch.bfloat16)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    cu_seq_lens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    prefill, lse = tokenspeed_mla_prefill(
        query=query,
        key=key,
        value=value,
        seq_lens=seq_lens,
        cum_seq_lens=cu_seq_lens,
        max_seq_len=seq_len,
        batch_size=1,
        softmax_scale=scale,
        is_causal=True,
        return_lse=True,
        enable_pdl=False,
    )
    torch.cuda.synchronize()
    prefill_ref = _attention_reference(query, key, value, scale, causal=True).to(
        prefill.dtype
    )
    prefill_error = _error(prefill, prefill_ref)
    torch.testing.assert_close(prefill, prefill_ref, atol=8.0e-2, rtol=8.0e-2)
    assert torch.isfinite(lse).all()

    batch, q_len, decode_heads = 1, 1, 16
    kv_lora_rank, rope_dim, page_size = 512, 64, 64
    head_dim = kv_lora_rank + rope_dim
    query_real = torch.randn(
        batch, q_len, decode_heads, head_dim, device=device, dtype=torch.bfloat16
    ) * 0.25
    cache_real = torch.randn(
        1, page_size, head_dim, device=device, dtype=torch.bfloat16
    ) * 0.25
    query_fp8 = query_real.to(torch.float8_e4m3fn)
    cache_fp8 = cache_real.to(torch.float8_e4m3fn)
    block_tables = torch.tensor([[0]], device=device, dtype=torch.int32)
    decode_seq_lens = torch.tensor([page_size], device=device, dtype=torch.int32)
    workspace_bytes = (
        get_num_sm(device) * decode_heads * 8 * (kv_lora_rank + 1) * 4
    )
    workspace = torch.empty(workspace_bytes, device=device, dtype=torch.int8)
    decoded = tokenspeed_mla_decode(
        query=query_fp8,
        kv_cache=cache_fp8,
        workspace_buffer=workspace,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_dim,
        block_tables=block_tables,
        seq_lens=decode_seq_lens,
        max_seq_len=page_size,
        softmax_scale=1.0 / math.sqrt(head_dim),
        output_scale=1.0,
        enable_pdl=False,
    )
    torch.cuda.synchronize()
    decode_ref = _attention_reference(
        query_fp8[0],
        cache_fp8[0, :, None, :],
        cache_fp8[0, :, None, :kv_lora_rank],
        1.0 / math.sqrt(head_dim),
        causal=False,
    ).to(decoded.dtype)
    decode_error = _error(decoded[0], decode_ref)
    torch.testing.assert_close(decoded[0], decode_ref, atol=1.2e-1, rtol=1.2e-1)
    return {
        "prefill": {
            "query_shape": list(query.shape),
            "output_shape": list(prefill.shape),
            "lse_shape": list(lse.shape),
            "error": prefill_error,
        },
        "decode": {
            "query_shape": list(query_fp8.shape),
            "cache_shape": list(cache_fp8.shape),
            "output_shape": list(decoded.shape),
            "workspace_bytes": workspace_bytes,
            "error": decode_error,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-vllm-version", required=True)
    args = parser.parse_args()

    packages = {
        name: version(name)
        for name in ("apache-tvm-ffi", "tilelang", "tokenspeed-mla", "vllm")
    }
    expected = {
        "apache-tvm-ffi": "0.1.12",
        "tilelang": "0.1.14",
        "tokenspeed-mla": "0.1.9",
        "vllm": args.expected_vllm_version,
    }
    if packages != expected:
        raise RuntimeError(f"package versions changed: {packages!r} != {expected!r}")

    capability = torch.cuda.get_device_capability()
    result: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": capability,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": packages,
        "tilelang_mhc": probe_tilelang_mhc(),
    }
    if capability[0] == 10:
        result["tokenspeed_mla"] = probe_tokenspeed_mla()
    else:
        result["tokenspeed_mla"] = {
            "skipped": "TokenSpeed MLA is a Blackwell-only backend"
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
