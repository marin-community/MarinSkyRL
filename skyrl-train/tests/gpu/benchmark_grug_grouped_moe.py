"""One-H100 Grug eager/grouped sparse-block benchmark, intentionally outside CI.

The defaults are the June step-630 Snowball 67B expert shape and one 8K policy
microbatch. The script reports warmed forward/backward time and peak allocated
HBM for the same block, weights, and input in both modes.

Run::

    python tests/gpu/benchmark_grug_grouped_moe.py
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeSparseMoeBlock


def _config(args: argparse.Namespace) -> GrugMoeConfig:
    return GrugMoeConfig(
        vocab_size=128256,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        shared_expert_intermediate_size=2560,
        num_local_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        num_hidden_layers=1,
        num_attention_heads=20,
        num_key_value_heads=5,
        head_dim=128,
        max_position_embeddings=args.tokens,
        sliding_window=2048,
        initializer_range=0.02,
        qk_mult=1.0,
    )


def _initialize(block: GrugMoeSparseMoeBlock) -> None:
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(mean=0.0, std=0.02)


def _build_block(config: GrugMoeConfig, device: torch.device) -> GrugMoeSparseMoeBlock:
    # Construct the 5 GiB expert stack directly in BF16 HBM. Building it first
    # in host FP32 would create an unnecessary 10 GiB staging allocation.
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            return GrugMoeSparseMoeBlock(config)
    finally:
        torch.set_default_dtype(original_dtype)


def _iteration(
    block: GrugMoeSparseMoeBlock,
    hidden_states: torch.Tensor,
) -> tuple[float, float]:
    block.zero_grad(set_to_none=True)
    hidden_states.grad = None
    start = torch.cuda.Event(enable_timing=True)
    forward_done = torch.cuda.Event(enable_timing=True)
    backward_done = torch.cuda.Event(enable_timing=True)
    start.record()
    output = block(hidden_states)
    forward_done.record()
    output.float().square().mean().backward()
    backward_done.record()
    backward_done.synchronize()
    forward_ms = start.elapsed_time(forward_done)
    total_ms = start.elapsed_time(backward_done)
    del output
    return forward_ms, total_ms - forward_ms


def _measure(
    block: GrugMoeSparseMoeBlock,
    hidden_states: torch.Tensor,
    *,
    grouped: bool,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | str]:
    block.experts.use_grouped_mm = grouped
    for _ in range(warmup):
        _iteration(block, hidden_states)
    block.zero_grad(set_to_none=True)
    hidden_states.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    forward_times = []
    backward_times = []
    for _ in range(iterations):
        forward_ms, backward_ms = _iteration(block, hidden_states)
        forward_times.append(forward_ms)
        backward_times.append(backward_ms)

    total_times = [forward + backward for forward, backward in zip(forward_times, backward_times)]
    return {
        "mode": "grouped" if grouped else "eager",
        "iterations": iterations,
        "forward_ms_median": statistics.median(forward_times),
        "backward_ms_median": statistics.median(backward_times),
        "forward_backward_ms_median": statistics.median(total_times),
        "tokens_per_second": hidden_states.shape[1] / (statistics.median(total_times) / 1000.0),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--intermediate-size", type=int, default=1280)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    if properties.major < 9:
        raise RuntimeError(f"native grouped-MM requires Hopper, found {properties.name}")

    torch.manual_seed(20260731)
    block = _build_block(_config(args), device)
    _initialize(block)
    hidden_states = torch.randn(
        1,
        args.tokens,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    eager = _measure(
        block,
        hidden_states,
        grouped=False,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    grouped = _measure(
        block,
        hidden_states,
        grouped=True,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    report = {
        "device": properties.name,
        "dtype": "bfloat16",
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "eager": eager,
        "grouped": grouped,
        "grouped_speedup": eager["forward_backward_ms_median"] / grouped["forward_backward_ms_median"],
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
