"""Values and gradients of one Snowball decoder layer under the vLLM-kernel numerics, with and without recompute.

For each layer and numerics set, on one GPU with the layer's real weights and a random input sequence:

- the forward with gradients equals the forward without gradients byte for byte (the probe scores
  without gradients; training computes the same values);
- a forward and backward inside Megatron's ``tensor_parallel.checkpoint`` (full activation recompute: the
  forward runs without gradients and again during the backward, router replay serving the recompute
  from its FIFO) gives the same output and gradients as the plain forward and backward;
- the gradients stay close to the trainer's default numerics (relative L2 distance of the input
  gradient and of every parameter gradient), since each flag keeps a default backward. Every run replays the
  experts the default run chose, so a near-tie routing flip cannot move a whole expert's gradient.

Example::

    python -m skyrl_train.mismatch_harness.grad_check --model s3://.../hf-bf16-vllm --layers 0 3 --tokens 512 \\
        --numerics compiled_stack=... --numerics vllm_kernel_stack=... --output s3://.../grad/<name>
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from marinskyrl.resource_locator import join_resource_path
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

HOME_RANK = 3
EP_SIZE = 8


def _relative(left: torch.Tensor, right: torch.Tensor) -> float:
    return ((left.float() - right.float()).norm() / right.float().norm().clamp_min(1e-30)).item()


def default_routes(layer, hidden, rotary_pos_emb) -> torch.Tensor:
    """The experts the layer's router picks for each token at default numerics, ``[tokens, top_k]``."""
    kept = {}
    handle = layer.mlp.router.register_forward_hook(lambda module, args, output: kept.update(map=output[1]))
    try:
        with torch.no_grad():
            clear_numerics_handoffs()
            layer(hidden_states=hidden, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
    finally:
        handle.remove()
        clear_numerics_handoffs()
    routing_map = kept["map"]
    return routing_map.nonzero()[:, 1].view(routing_map.shape[0], layer.mlp.router.topk)


def run_layer(layer, hidden, grad, rotary_pos_emb, flags, routes, *, recompute: bool, with_grad: bool = True):
    """Output, input gradient and parameter gradients of one forward (and backward) of the layer on ``routes``."""
    router = layer.mlp.router
    tokens = hidden.shape[0] * hidden.shape[1]
    controller = MegatronRouterReplay([0], recompute_enabled=recompute)
    router.router_replay = LayerReplayHandle(controller, 0)
    controller.begin_forward(
        {0: routes},
        torch.ones(tokens, dtype=torch.bool, device="cuda"),
        record_recompute=recompute,
        vllm_expert_parallel=VllmExpertParallel(torch.full((tokens,), HOME_RANK, device="cuda"), EP_SIZE),
    )
    layer.zero_grad(set_to_none=True)
    source = hidden.detach().clone().requires_grad_(with_grad)

    def forward(inputs):
        clear_numerics_handoffs()
        output, _ = layer(hidden_states=inputs, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
        return output

    try:
        with grug_numerics(**flags), torch.set_grad_enabled(with_grad):
            output = tensor_parallel.checkpoint(forward, False, source) if recompute else forward(source)
            controller.end_forward()
            if with_grad:
                output.backward(grad)
    finally:
        router.router_replay = None
        clear_numerics_handoffs()
    controller.assert_drained()
    parameters = {
        name: param.grad.detach().clone() for name, param in layer.named_parameters() if param.grad is not None
    }
    return output.detach(), (source.grad.detach().clone() if with_grad else None), parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--numerics", action="append", default=[], help="[label=]comma-separated flags")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    results: dict = {"arguments": vars(args), "layers": {}}
    with tempfile.TemporaryDirectory() as directory, single_rank_megatron(args.seed), vllm_config_context():
        config_dir = Path(directory)
        shape = GrugShape.from_config(stage_config(args.model, config_dir))
        _, provider = grug_provider(str(config_dir))
        names = sorted({name for layer in args.layers for name in layer_weight_names(layer, experts=True)})
        weights = export_weights(args.model, config_dir, names)
        rotary = rotary_embedding(provider).cuda()
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        hidden = torch.randn(args.tokens, 1, shape.hidden, generator=generator, device="cuda").to(torch.bfloat16)
        grad = torch.randn(args.tokens, 1, shape.hidden, generator=generator, device="cuda").to(torch.bfloat16)
        variants = parse_variants(args.numerics)
        for index in args.layers:
            layer = build_layer(provider, index)
            load_hf_weights(layer, f"decoder.layers.{index}.", weights)
            layer.train()
            rotary_pos_emb = rotary(args.tokens)
            routes = default_routes(layer, hidden, rotary_pos_emb)
            reference = run_layer(layer, hidden, grad, rotary_pos_emb, {}, routes, recompute=False)
            entry = {}
            for label, flags in variants.items():
                plain = run_layer(layer, hidden, grad, rotary_pos_emb, flags, routes, recompute=False)
                scored = run_layer(layer, hidden, grad, rotary_pos_emb, flags, routes, recompute=False, with_grad=False)
                checkpointed = run_layer(layer, hidden, grad, rotary_pos_emb, flags, routes, recompute=True)
                relative = {name: _relative(plain[2][name], reference[2][name]) for name in reference[2]}
                worst = max(relative, key=relative.get)
                entry[label] = {
                    "no_grad_output_equal": bool(torch.equal(plain[0], scored[0])),
                    "recompute_output_equal": bool(torch.equal(plain[0], checkpointed[0])),
                    "recompute_input_grad_relative": _relative(checkpointed[1], plain[1]),
                    "recompute_param_grad_max_relative": max(
                        _relative(checkpointed[2][name], plain[2][name]) for name in plain[2]
                    ),
                    "input_grad_relative_to_default": _relative(plain[1], reference[1]),
                    "param_grad_max_relative_to_default": relative[worst],
                    "param_grad_max_relative_name": worst,
                    "param_grad_median_relative_to_default": sorted(relative.values())[len(relative) // 2],
                    "output_byte_equal_to_default": (plain[0].view(torch.int16) == reference[0].view(torch.int16))
                    .float()
                    .mean()
                    .item(),
                    "grads_finite": bool(
                        torch.isfinite(plain[1]).all() and all(torch.isfinite(g).all() for g in plain[2].values())
                    ),
                    "parameters_with_grad": len(plain[2]),
                }
                print(json.dumps({"layer": index, "variant": label, **entry[label]}), flush=True)
            results["layers"][str(index)] = entry
            del layer
            torch.cuda.empty_cache()
    payload = json.dumps(results, indent=1, sort_keys=True)
    io.write_bytes_atomic(join_resource_path(args.output, "grad_check.json"), payload.encode())
    print(payload, flush=True)
    print("PASS mismatch harness grad check", args.output, flush=True)


if __name__ == "__main__":
    main()
