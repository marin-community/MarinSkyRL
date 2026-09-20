"""Small, repeatable Transformer Engine attention probe for FA2/FA4 arms.

Run in separate processes with NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 and
NVTE_FUSED_ATTN=0. Inspect the Transformer Engine backend-selection log;
package presence alone is not evidence that a kernel was selected.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from importlib import metadata
from pathlib import Path

import torch
from transformer_engine.pytorch.attention import DotProductAttention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--kv-heads", type=int, default=5)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window-left", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    args = parse_args()
    required_environment = {
        "NVTE_DEBUG": "1",
        "NVTE_DEBUG_LEVEL": "2",
        "NVTE_FUSED_ATTN": "0",
        "NVTE_FLASH_ATTN": "1",
    }
    for name, value in required_environment.items():
        if os.getenv(name) != value:
            raise ValueError(f"Set {name}={value} so the backend choice is visible and comparable")
    if os.getenv("NVTE_FLASH_ATTN_V4") not in ("0", "1"):
        raise ValueError("Set NVTE_FLASH_ATTN_V4=0 for FA2 or 1 for FA4")
    torch.manual_seed(1704)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    shape = (args.batch, args.seq)
    q = torch.randn(*shape, args.heads, args.head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(*shape, args.kv_heads, args.head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(*shape, args.kv_heads, args.head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn(*shape, args.heads * args.head_dim, device=device, dtype=torch.bfloat16)
    attention = DotProductAttention(
        num_attention_heads=args.heads,
        num_gqa_groups=args.kv_heads,
        kv_channels=args.head_dim,
        qkv_format="bshd",
        attn_mask_type="causal",
        window_size=(args.window_left, 0),
        attention_dropout=0.0,
    ).to(device)

    def step() -> tuple[torch.Tensor, float, float]:
        for tensor in (q, k, v):
            tensor.grad = None
        start = time.perf_counter()
        output = attention(q, k, v)
        torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - start) * 1000
        loss = (output.float() * target.float()).sum() / output.numel()
        loss.backward()
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - start) * 1000
        return output, forward_ms, total_ms

    for _ in range(args.warmups):
        step()
    torch.cuda.reset_peak_memory_stats()
    forward_ms = []
    total_ms = []
    for _ in range(args.samples):
        output, forward, total = step()
        forward_ms.append(forward)
        total_ms.append(total)

    result = {
        "shape": {
            "batch": args.batch,
            "seq": args.seq,
            "heads": args.heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "window_left": args.window_left,
        },
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "packages": {
            name: package_version(name)
            for name in ("transformer-engine", "transformer-engine-torch", "flash-attn", "flash-attn-4")
        },
        "environment": required_environment | {"NVTE_FLASH_ATTN_V4": os.environ["NVTE_FLASH_ATTN_V4"]},
        "forward_ms": forward_ms,
        "total_ms": total_ms,
        "forward_median_ms": statistics.median(forward_ms),
        "total_median_ms": statistics.median(total_ms),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "finite": all(torch.isfinite(t).all().item() for t in (output, q.grad, k.grad, v.grad)),
    }
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "result": result,
            "output": output.float().cpu(),
            "dq": q.grad.float().cpu(),
            "dk": k.grad.float().cpu(),
            "dv": v.grad.float().cpu(),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
