"""H100 time of the trainer's attention and MoE combine under the vLLM-kernel numerics, at Snowball shapes.

Runs on one GPU, gradients on, as a training forward and backward runs them (no activation recompute; the
production recipe's full recompute runs each forward a second time before the backward):

- core attention alone, Megatron's TE ``DotProductAttention`` with the probe's settings (cuDNN), against
  the ``fa3_attention`` path (vLLM's FA3 forward, and the cuDNN forward and backward for the gradient), and
  against FA3's forward with FA2's backward, the other backward available in the runtime;
- one whole Grug decoder layer (256 experts on this GPU) under numerics sets, ``ep_sum`` included.

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


def layer_times(layer, rotary, shape: GrugShape, tokens: int, variants, generator: torch.Generator) -> dict[str, float]:
    """Forward+backward milliseconds of the whole layer on one sequence, per numerics variant."""
    layer.train()
    hidden = torch.randn(tokens, 1, shape.hidden, generator=generator, device="cuda").to(torch.bfloat16)
    hidden.requires_grad_()
    grad = torch.randn_like(hidden)
    rotary_pos_emb = rotary(tokens)
    router = layer.mlp.router
    times = {}
    for label, flags in variants.items():
        controller = None
        if flags.get("ep_sum"):
            controller = MegatronRouterReplay([0], recompute_enabled=False)
            router.router_replay = LayerReplayHandle(controller, 0)

        def run() -> None:
            if controller is not None:
                controller.begin_forward(
                    {0: torch.zeros(tokens, router.topk, dtype=torch.long, device="cuda")},
                    torch.zeros(tokens, dtype=torch.bool, device="cuda"),
                    vllm_expert_parallel=VllmExpertParallel(torch.zeros(tokens, dtype=torch.long, device="cuda"), 8),
                )
            clear_numerics_handoffs()
            output, _ = layer(hidden_states=hidden, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
            if controller is not None:
                controller.end_forward()
            output.backward(grad)
            clear_numerics_handoffs()

        with grug_numerics(**flags):
            times[label] = milliseconds(run)
        router.router_replay = None
    layer.eval()
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--numerics", action="append", default=[], help="[label=]comma-separated flags")
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
            entry = {"window": layer.self_attention.fa3_window, "attention": {}, "layer": {}}
            for tokens in args.lengths:
                entry["attention"][str(tokens)] = attention_times(layer, shape, tokens, generator)
                entry["layer"][str(tokens)] = layer_times(layer, rotary, shape, tokens, variants, generator)
                print(
                    json.dumps({"layer": layer_index, "tokens": tokens, **entry["attention"][str(tokens)]}), flush=True
                )
                print(json.dumps({"layer": layer_index, "tokens": tokens, **entry["layer"][str(tokens)]}), flush=True)
            results["layers"][str(layer_index)] = entry
            del layer
            torch.cuda.empty_cache()
    payload = json.dumps(results, indent=1, sort_keys=True)
    io.write_bytes_atomic(join_resource_path(args.output, "timing.json"), payload.encode())
    print(payload, flush=True)
    print("PASS mismatch harness timing", args.output, flush=True)


if __name__ == "__main__":
    main()
