"""H100 time of the trainer's attention, experts and whole layer under the vLLM-kernel numerics, at Snowball shapes.

Runs on one GPU, gradients on, as a training forward and backward runs them:

- core attention alone, Megatron's TE ``DotProductAttention`` with the probe's settings (cuDNN), against
  the ``fa3_attention`` path (vLLM's FA3 forward, and the cuDNN forward and backward for the gradient), and
  against FA3's forward with FA2's backward, the other backward available in the runtime;
- one whole Grug decoder layer under numerics sets, forward and backward, both plain and inside Megatron's
  ``tensor_parallel.checkpoint`` (the production recipe's full recompute: a forward without gradients, then the
  forward again with gradients and the backward). The experts either route natively over all 256 experts on
  this GPU (``native``), or every token is routed among the first 32 experts, which gives this GPU the expert
  rows one of 8 expert-parallel ranks computes (``ep8_rank``);
- the routed experts' forward alone without gradients (the scoring and first recompute pass), Transformer
  Engine's grouped GEMMs against vLLM's fused-MoE kernels, at both expert loads.

Example::

    python -m skyrl_train.mismatch_harness.timing --model s3://.../hf-bf16-vllm --layers 0 3 \\
        --lengths 512 1024 2048 4096 8192 --output s3://.../timing/<name>
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch
from flash_attn.flash_attn_interface import _flash_attn_varlen_backward
from marinskyrl.resource_locator import join_resource_path
from megatron.core.transformer.enums import AttnMaskType
from vllm.vllm_flash_attn import flash_attn_varlen_func

from megatron.core import tensor_parallel

from skyrl_train.io import io
from skyrl_train.mismatch_harness.run import export_weights, layer_weight_names, parse_variants, stage_config
from skyrl_train.mismatch_harness.trainer_side import (
    build_layer,
    grug_provider,
    load_hf_weights,
    rotary_embedding,
    single_rank_megatron,
)
from skyrl_train.mismatch_harness.vllm_side import GrugShape, vllm_config_context
from skyrl_train.mismatch_probe.numerics import grug_numerics
from skyrl_train.models.grug_megatron import clear_numerics_handoffs
from skyrl_train.models.megatron_router_replay import LayerReplayHandle, MegatronRouterReplay, VllmExpertParallel

WARMUP = 3
REPETITIONS = 10


def milliseconds(run: Callable[[], None]) -> float:
    """Median CUDA time of ``run`` over ``REPETITIONS`` calls after ``WARMUP`` calls."""
    for _ in range(WARMUP):
        run()
    times = []
    for _ in range(REPETITIONS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def attention_times(layer, shape: GrugShape, tokens: int, generator: torch.Generator) -> dict[str, float]:
    """Forward and forward+backward milliseconds of one layer's core attention on one sequence."""
    core = layer.self_attention.core_attention
    attention = layer.self_attention

    def tensor(heads: int) -> torch.Tensor:
        value = torch.randn(tokens, 1, heads, shape.head_dim, generator=generator, device="cuda")
        return value.to(torch.bfloat16).requires_grad_()

    query, key, value = tensor(shape.heads), tensor(shape.kv_heads), tensor(shape.kv_heads)
    grad = torch.randn(tokens, 1, shape.heads * shape.head_dim, generator=generator, device="cuda").to(torch.bfloat16)

    def call() -> torch.Tensor:
        return core(query, key, value, None, attn_mask_type=AttnMaskType.causal)

    def forward_backward() -> None:
        call().backward(grad)

    times = {}
    with torch.no_grad():
        times["cudnn_forward"] = milliseconds(call)
    times["cudnn_forward_backward"] = milliseconds(forward_backward)
    with grug_numerics(fa3_attention=True):
        with torch.no_grad():
            times["fa3_forward"] = milliseconds(call)
        times["fa3_attention_forward_backward"] = milliseconds(forward_backward)
    times["fa3_forward_fa2_backward"] = _fa3_fa2_time(query, key, value, grad, attention, shape)
    return times


def _fa3_fa2_time(query, key, value, grad, attention, shape: GrugShape) -> float:
    """FA3's forward (vLLM's build) with FA2's backward on FA3's output and log-sum-exp."""
    tokens = query.shape[0]
    q, k, v = (tensor.detach().squeeze(1).contiguous() for tensor in (query, key, value))
    dout = grad.view(tokens, shape.heads, shape.head_dim).contiguous()
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    window = attention.fa3_window
    left = -1 if window is None else window - 1

    def run() -> None:
        out, lse = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=tokens,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=tokens,
            cu_seqlens_k=cu_seqlens,
            softmax_scale=attention.fa3_scale,
            causal=True,
            window_size=None if window is None else [left, 0],
            fa_version=3,
            num_splits=1,
            return_softmax_lse=True,
        )
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_varlen_backward(
            dout, q, k, v, out, lse, dq, dk, dv, cu_seqlens, cu_seqlens, tokens, tokens, 0.0,
            attention.fa3_scale, True, left, 0, 0.0, None, False,
        )  # fmt: skip

    return milliseconds(run)


# Experts one of vLLM's and the trainer's 8 expert-parallel ranks holds (256 experts over 8 ranks).
EP8_RANK_EXPERTS = 32


def expert_routes(tokens: int, top_k: int, experts: int, generator: torch.Generator) -> torch.Tensor:
    """``[tokens, top_k]`` distinct experts per token drawn uniformly from the first ``experts``."""
    scores = torch.rand(tokens, experts, generator=generator, device="cuda")
    return scores.argsort(dim=1)[:, :top_k]


def layer_times(
    layer, rotary, shape: GrugShape, tokens: int, variants, generator: torch.Generator
) -> dict[str, dict[str, float]]:
    """Forward+backward milliseconds of the whole layer on one sequence, per numerics variant, expert load and
    recompute setting.

    ``native`` routes with the layer's router; ``ep8_rank`` replays routes among the first 32 experts. The
    ``recompute`` times run the layer inside ``tensor_parallel.checkpoint`` as the production recipe does.
    """
    layer.train()
    hidden = torch.randn(tokens, 1, shape.hidden, generator=generator, device="cuda").to(torch.bfloat16)
    grad = torch.randn_like(hidden)
    rotary_pos_emb = rotary(tokens)
    router = layer.mlp.router
    loads = {
        "native": (torch.zeros(tokens, router.topk, dtype=torch.long, device="cuda"), False),
        "ep8_rank": (expert_routes(tokens, router.topk, EP8_RANK_EXPERTS, generator), True),
    }
    times: dict[str, dict[str, float]] = {}
    for label, flags in variants.items():
        for load, (routes, replayed) in loads.items():
            for recompute in (False, True):
                controller = MegatronRouterReplay([0], recompute_enabled=recompute)
                router.router_replay = LayerReplayHandle(controller, 0)

                def forward(inputs):
                    clear_numerics_handoffs()
                    output, _ = layer(hidden_states=inputs, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
                    return output

                def run() -> None:
                    controller.begin_forward(
                        {0: routes},
                        torch.full((tokens,), replayed, dtype=torch.bool, device="cuda"),
                        record_recompute=recompute,
                        vllm_expert_parallel=VllmExpertParallel(
                            torch.zeros(tokens, dtype=torch.long, device="cuda"), 8
                        ),
                    )
                    source = hidden.detach().requires_grad_()
                    output = tensor_parallel.checkpoint(forward, False, source) if recompute else forward(source)
                    controller.end_forward()
                    output.backward(grad)
                    clear_numerics_handoffs()

                with grug_numerics(**flags):
                    times.setdefault(label, {})[f"{load}{'+recompute' if recompute else ''}"] = milliseconds(run)
                controller.assert_drained()
                router.router_replay = None
    layer.eval()
    return times


def expert_forward_times(
    layer, shape: GrugShape, tokens: int, generator: torch.Generator
) -> dict[str, dict[str, float]]:
    """Milliseconds of the routed experts' forward without gradients: TE's grouped GEMMs and vLLM's kernels."""
    experts = layer.mlp.experts
    hidden = torch.randn(tokens, shape.hidden, generator=generator, device="cuda").to(torch.bfloat16)
    times: dict[str, dict[str, float]] = {}
    for load, count in (("native", shape.experts), ("ep8_rank", EP8_RANK_EXPERTS)):
        routes = expert_routes(tokens, shape.top_k, count, generator)
        routing_map = torch.zeros(tokens, shape.experts, dtype=torch.bool, device="cuda").scatter(1, routes, True)
        pairs = routing_map.t().nonzero()
        permuted = hidden[pairs[:, 1]]
        probs = torch.rand(pairs.shape[0], generator=generator, device="cuda")
        counts = routing_map.sum(0).cpu()
        for label, flags in (("te_grouped_gemm", {}), ("vllm_experts", {"vllm_experts": True})):

            def run() -> None:
                experts(permuted, counts, probs)

            with torch.no_grad(), grug_numerics(**flags):
                times.setdefault(load, {})[label] = milliseconds(run)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--numerics", action="append", default=[], help="[label=]comma-separated flags")
    parser.add_argument("--skip-attention", action="store_true", help="time only the layer and the experts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    results: dict = {"arguments": vars(args), "gpu": torch.cuda.get_device_name(), "layers": {}}
    with tempfile.TemporaryDirectory() as directory, single_rank_megatron(args.seed), vllm_config_context():
        config_dir = Path(directory)
        shape = GrugShape.from_config(stage_config(args.model, config_dir))
        _, provider = grug_provider(str(config_dir))
        names = sorted({name for layer in args.layers for name in layer_weight_names(layer, experts=True)})
        weights = export_weights(args.model, config_dir, names)
        rotary = rotary_embedding(provider).cuda()
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        variants = parse_variants(args.numerics)
        for layer_index in args.layers:
            layer = build_layer(provider, layer_index)
            load_hf_weights(layer, f"decoder.layers.{layer_index}.", weights)
            entry = {"window": layer.self_attention.fa3_window, "attention": {}, "layer": {}, "experts": {}}
            for tokens in args.lengths:
                if not args.skip_attention:
                    entry["attention"][str(tokens)] = attention_times(layer, shape, tokens, generator)
                entry["experts"][str(tokens)] = expert_forward_times(layer, shape, tokens, generator)
                entry["layer"][str(tokens)] = layer_times(layer, rotary, shape, tokens, variants, generator)
                for part in ("attention", "experts", "layer"):
                    if str(tokens) in entry[part]:
                        record = {"layer": layer_index, "tokens": tokens, part: entry[part][str(tokens)]}
                        print(json.dumps(record), flush=True)
            results["layers"][str(layer_index)] = entry
            del layer
            torch.cuda.empty_cache()
    payload = json.dumps(results, indent=1, sort_keys=True)
    io.write_bytes_atomic(join_resource_path(args.output, "timing.json"), payload.encode())
    print(payload, flush=True)
    print("PASS mismatch harness timing", args.output, flush=True)


if __name__ == "__main__":
    main()
