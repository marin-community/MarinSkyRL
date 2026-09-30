"""GPU entry point: compare trainer-captured Grug layers with compiled vLLM's archived kernels.

Example (one H100)::

    python -m skyrl_train.mismatch_harness.run \\
        --capture s3://.../mismatch_probe-trainer-capture/update-0/native_capture/pp-0.pt \\
        --output-code s3://.../mismatch_probe-inductor-output-code/engine-0/dp-0-ep-0 \\
        --model s3://.../hf-bf16-vllm --layers 0 3 --ep-size 8 --output s3://.../harness/<name>

Weights come from a Hugging Face export (``--model``) or, for a probe that resumed from a training
checkpoint, from that Megatron checkpoint (``--megatron-checkpoint``) with ``--model`` naming the
export that supplies the config. The run writes ``harness.json`` and ``harness.md``.
"""

from __future__ import annotations

import argparse
import gc
import io as stdlib_io
import json
import platform
import tempfile
from pathlib import Path

import torch
from marinskyrl.resource_locator import join_resource_path, relative_resource_path
from torch._inductor.runtime.cache_dir_utils import cache_dir
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper

from skyrl_train.distributed.megatron.checkpoint_metadata import remote_checkpoint_metadata
from skyrl_train.distributed.megatron.direct_checkpoint import DirectS3TorchDistLoadShardedStrategy
from skyrl_train.io import io
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore
from skyrl_train.mismatch_harness.expert_parallel import ReduceOrder, reduction_order
from skyrl_train.mismatch_harness.harness import (
    LayerReplay,
    RowLayout,
    attention_inputs,
    compare_regions,
    emulated_ep_sum,
    piece_regions,
    routing_agreement,
    trainer_expert_slots,
    trainer_region_tensors,
    trainer_routing,
)
from skyrl_train.mismatch_harness.numerics import compare
from skyrl_train.mismatch_harness.output_code import is_graph_module, parse_output_code
from skyrl_train.mismatch_harness.pieces import classify
from skyrl_train.mismatch_harness.regions import POST_ATTENTION_ROLES, PRE_ATTENTION_ROLES
from skyrl_train.mismatch_harness.replay import load_module
from skyrl_train.mismatch_harness.report import render_markdown
from skyrl_train.mismatch_harness.trainer_side import (
    build_gated_norm,
    build_layer,
    gated_norm_regions,
    grug_provider,
    load_hf_weights,
    rotary_embedding,
    run_layer,
    single_rank_megatron,
)
from skyrl_train.mismatch_harness.vllm_side import (
    GrugShape,
    Requests,
    cos_sin_cache,
    cuda_graph_padded_tokens,
    vllm_config_context,
)

CONFIG_FILES = ("config.json", "model.safetensors.index.json")
CAPTURE_REGIONS = (
    "input",
    "attention_norm",
    "attention",
    "residual_after_attention",
    "mlp_norm",
    "router_probs",
    "router_map",
    "shared_expert",
    "mlp",
    "output",
)


def stage_config(model_uri: str, directory: Path) -> dict:
    for name in CONFIG_FILES:
        (directory / name).write_bytes(io.read_bytes(join_resource_path(model_uri, name)))
    return json.loads((directory / "config.json").read_text())


def load_captures(uris: list[str]) -> tuple[dict[int, dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
    layers: dict[int, dict[str, torch.Tensor]] = {}
    sequences = attention_mask = None
    for uri in uris:
        payload = torch.load(stdlib_io.BytesIO(io.read_bytes(uri)), map_location="cpu", weights_only=False)
        layers.update(payload["layers"])
        if sequences is not None and not torch.equal(sequences, payload["sequences"]):
            raise ValueError(f"{uri} holds a different micro-batch than the other captures")
        sequences, attention_mask = payload["sequences"], payload["attention_mask"]
    return layers, sequences, attention_mask


def stage_autotune_choices(output_code_uri: str) -> int:
    """Copy the worker's archived ``*.best_config`` files into this process's Inductor cache.

    A kernel with several candidate launch configs benchmarks them at its first launch and keeps the
    fastest in ``<kernel>.best_config`` beside its source; with the file in place the replay launches
    the config vLLM chose. Returns how many were staged.
    """
    root = Path(cache_dir())
    staged = 0
    for path in io.find_files(output_code_uri):
        if not path.endswith(".best_config"):
            continue
        relative = relative_resource_path(output_code_uri, path)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(io.read_bytes(join_resource_path(output_code_uri, relative)))
        staged += 1
    return staged


def kernel_candidates(pieces: dict) -> dict[tuple[str, str], list]:
    """Each loaded Triton kernel's compiled launch configs, taken before any launch autotunes it."""
    return {
        (f"{kind}{'+rope' if rope else ''}", name): list(getattr(namespace.get(name), "launchers", None) or [])
        for (kind, rope), (piece, namespace) in pieces.items()
        for name in piece.code.kernels
    }


def chosen_configs(pieces: dict, candidates: dict[tuple[str, str], list]) -> dict[str, dict[str, dict]]:
    """For each replayed Triton kernel, its candidate launch configs and the one it ran with."""
    chosen: dict[str, dict[str, dict]] = {}
    for (kind, rope), (piece, namespace) in pieces.items():
        key = f"{kind}{'+rope' if rope else ''}"
        chosen[key] = {
            name: {
                "candidates": [str(launcher.config) for launcher in candidates[(key, name)]],
                "launched": [
                    str(launcher.config) for launcher in getattr(namespace.get(name), "launchers", None) or []
                ],
            }
            for name in piece.code.kernels
        }
    return chosen


def load_pieces(output_code_uri: str, candidate: int | None = None) -> dict:
    """Parse, classify and load every compiled subgraph module of one vLLM worker.

    With ``candidate``, every kernel that has several launch configs keeps only that one (modulo the
    count), so no autotuning happens and the replay launches a config a vLLM process may have chosen.
    """
    pieces = {}
    for path in sorted(io.find_files(output_code_uri)):
        if not path.endswith(".py"):
            continue
        text = io.read_bytes(
            join_resource_path(output_code_uri, relative_resource_path(output_code_uri, path))
        ).decode()
        if not is_graph_module(text):
            continue
        piece = classify(parse_output_code(text))
        key = (piece.kind, piece.rope)
        if key in pieces:
            raise ValueError(f"two archived modules are {piece.kind} pieces with rope={piece.rope}")
        module = load_module(piece.code, f"vllm_piece_{piece.kind}_{int(piece.rope)}_{candidate}")
        if candidate is not None:
            for name in piece.code.kernels:
                kernel = module.__dict__[name]
                kernel.launchers = [kernel.launchers[candidate % len(kernel.launchers)]]
        pieces[key] = (piece, module.__dict__)
    return pieces


def needed_weights(shape: GrugShape, layers: list[int]) -> list[str]:
    names = {"model.embed_tokens.weight", "model.embed_norm.weight", "model.norm.weight"}
    names |= {
        f"model.{norm}.{proj}.weight"
        for norm in ("embed_gated_norm", "final_gated_norm")
        for proj in ("down_proj", "up_proj")
    }
    for layer in layers:
        names |= set(layer_weight_names(layer, experts=True))
        if layer + 1 < shape.layers:
            names |= set(layer_weight_names(layer + 1, experts=False))
    return sorted(names)


def layer_weight_names(layer: int, *, experts: bool) -> list[str]:
    prefix = f"model.layers.{layer}."
    names = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        *(
            f"{norm}.{proj}.weight"
            for norm in ("attn_gated_norm", "mlp_gated_norm")
            for proj in ("down_proj", "up_proj")
        ),
        *(f"self_attn.{proj}.weight" for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "attn_gate")),
        "mlp.router.weight",
        "mlp.router.bias",
        *(f"shared_expert.{proj}.weight" for proj in ("gate_proj", "up_proj", "down_proj")),
    ]
    if experts:
        names += [f"mlp.experts.{proj}.weight" for proj in ("gate_proj", "up_proj", "down_proj")]
    return [prefix + name for name in names]


def export_weights(model_uri: str, config_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    store = RemoteSafetensorsTensorStore(model_uri, config_dir)
    return {name: tensor.cuda() for name, tensor in store.load_tensors(names).items()}


def checkpoint_weights(bridge, provider, checkpoint_uri: str, names: list[str]) -> dict[str, torch.Tensor]:
    """Load a SkyRL Megatron policy checkpoint into the full model and export Hugging Face tensors."""
    model = provider.provide_distributed_model(wrap_with_ddp=False, bf16=True)
    unwrapped = model[0].module if hasattr(model[0], "module") else model[0]
    sharded = {"model": unwrapped.sharded_state_dict()}
    strategy = FullyParallelLoadStrategyWrapper(
        DirectS3TorchDistLoadShardedStrategy(checkpoint_uri),
        parallel_state.get_data_parallel_group(with_context_parallel=True),
    )
    with remote_checkpoint_metadata(checkpoint_uri) as read_dir:
        state = dist_checkpointing.load(sharded_state_dict=sharded, checkpoint_dir=read_dir, sharded_strategy=strategy)
    model[0].load_state_dict(state["model"], strict=True)
    wanted = set(names)
    tensors = {
        name: tensor.cuda()
        for name, tensor in bridge.export_hf_weights(model, cpu=False, show_progress=False)
        if name in wanted
    }
    missing = wanted - set(tensors)
    if missing:
        raise KeyError(f"the checkpoint export lacks {sorted(missing)}")
    del model
    torch.cuda.empty_cache()
    return tensors


def run_vllm(
    replay: LayerReplay,
    layout: RowLayout,
    shape: GrugShape,
    trainer: dict[str, torch.Tensor] | None,
    raw_trainer: dict[str, torch.Tensor],
    layer_input: torch.Tensor,
    *,
    config_tokens: int | None,
    pad_tokens: int | None,
    extra_input: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """One vLLM pass over layer ``L``: ``trainer`` given means isolated (every input from the trainer).

    The compiled pieces run on ``pad_tokens`` rows (default: the CUDA-graph size vLLM pads the step to).
    """
    lengths = layout.lengths if extra_input is None else (*layout.lengths, extra_input.shape[0])
    total = sum(lengths)
    requests = Requests(lengths, padded=pad_tokens or cuda_graph_padded_tokens(total))
    if extra_input is not None:
        layer_input = torch.cat((layer_input, extra_input))
    tokens = layout.requests.tokens
    pre, pre_piece = replay.run_pre_attention(requests, layer_input, None, trainer)
    regions = piece_regions(pre, pre_piece, PRE_ATTENTION_ROLES)
    inputs = attention_inputs(pre, shape)
    regions["query"], regions["key"] = inputs["query"], inputs["key"]
    if trainer is None:
        query, key, value = inputs["query"], inputs["key"], inputs["value"]
    else:
        query, key = trainer["query"], trainer["key"]
        value = trainer["v_proj"].reshape(tokens, shape.kv_heads, shape.head_dim)
    attention = replay.attention(query, key, value, requests)
    regions["core_attention"] = attention
    routing = None
    if trainer is not None:
        routing = trainer_routing(raw_trainer, layout, replay.router_bias, shape.top_k)
    moe = replay.moe(config_tokens, routing)
    if trainer is None:
        post_inputs = (attention, value.reshape(value.shape[0], -1), regions["attention_norm"])
    else:
        post_inputs = (trainer["core_attention"], trainer["v_proj"], trainer["attention_norm"])
    post, post_piece = replay.run_post_attention(requests, *post_inputs, layer_input, moe, trainer)
    regions.update(piece_regions(post, post_piece, POST_ATTENTION_ROLES))
    # Inductor pads the attn_gate GEMM to 24 output columns; the heads are the leading ones.
    regions["attn_gate"] = regions["attn_gate"][:, : shape.heads]
    record = moe.records[-1]
    regions["routing"] = (record.vllm_ids[:tokens], record.vllm_weights[:tokens])
    if trainer is not None:
        ids, _ = routing
        regions["expert_slots"] = record.slots
        trainer_slots = trainer_expert_slots(raw_trainer, layout, ids)
        order = reduction_order(replay.order, replay.ep_size, replay.home_rank)
        regions["ep_sum"] = emulated_ep_sum(trainer_slots, ids, shape.experts, replay.ep_size, order)
        regions["trainer_expert_slots"] = trainer_slots
    return {name: value[:tokens] if isinstance(value, torch.Tensor) else value for name, value in regions.items()}


def stats_json(stats: dict) -> dict:
    return {region: value.to_json() for region, value in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", action="append", required=True)
    parser.add_argument("--output-code", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--megatron-checkpoint")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--home-rank", type=int, default=0)
    parser.add_argument(
        "--reduce-order", choices=[order.value for order in ReduceOrder], default=ReduceOrder.RING.value
    )
    parser.add_argument("--moe-config-tokens", type=int)
    parser.add_argument(
        "--pad-tokens", type=int, help="rows the compiled pieces run on (default: vLLM's CUDA-graph size)"
    )
    parser.add_argument("--floor-extra-tokens", type=int, default=512)
    parser.add_argument("--numerics", action="append", default=[], help="comma-separated grug_numerics flags")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    order = ReduceOrder(args.reduce_order)
    results: dict = {
        "arguments": vars(args),
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "python": platform.python_version(),
        },
        "layers": {},
    }
    with tempfile.TemporaryDirectory() as directory, single_rank_megatron(args.seed), vllm_config_context():
        config_dir = Path(directory)
        config = stage_config(args.model, config_dir)
        shape = GrugShape.from_config(config)
        captured, sequences, attention_mask = load_captures(args.capture)
        layout = RowLayout(tuple(int(length) for length in attention_mask.sum(dim=1).tolist()))
        results["staged_best_configs"] = stage_autotune_choices(args.output_code)
        pieces = load_pieces(args.output_code)
        candidates = kernel_candidates(pieces)
        most = max(len(options) for options in candidates.values())
        pinned_pieces = [load_pieces(args.output_code, candidate=index) for index in range(most)] if most > 1 else []
        results["pieces"] = sorted(f"{kind}{'+rope' if rope else ''}" for kind, rope in pieces)
        bridge, provider = grug_provider(str(config_dir))
        names = needed_weights(shape, args.layers)
        if args.megatron_checkpoint:
            weights = checkpoint_weights(bridge, provider, args.megatron_checkpoint, names)
        else:
            weights = export_weights(args.model, config_dir, names)
        rotary = rotary_embedding(provider).cuda()
        cos_sin = cos_sin_cache(shape)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        for layer in args.layers:
            regions = captured[layer]
            layer_module = build_layer(provider, layer)
            load_hf_weights(layer_module, f"decoder.layers.{layer}.", weights)
            layer_input = regions["input"].cuda()
            next_norm = build_gated_norm(provider)
            if layer + 1 < shape.layers:
                load_hf_weights(next_norm, f"decoder.layers.{layer + 1}.input_layernorm.", weights)
            else:
                load_hf_weights(next_norm, "decoder.final_layernorm.", weights)
            baseline = run_layer(layer_module, layer_input, rotary, next_norm=next_norm, numerics={})
            reproduction = {}
            for region in CAPTURE_REGIONS:
                ours, theirs = baseline.tensors[region], regions[region].cuda()
                if ours.dtype == torch.bool:
                    reproduction[region] = {"equal_fraction": (ours == theirs).float().mean().item()}
                else:
                    reproduction[region] = compare(
                        ours.reshape(ours.shape[0], -1), theirs.reshape(theirs.shape[0], -1)
                    ).to_json()
            bias = layer_module.mlp.router.expert_bias.detach().float().clone()
            export_bias = weights[f"model.layers.{layer}.mlp.router.bias"].float()
            replay = LayerReplay(
                pieces=pieces,
                shape=shape,
                weights=weights,
                cos_sin=cos_sin,
                layer=layer,
                router_bias=bias,
                ep_size=args.ep_size,
                home_rank=args.home_rank,
                order=order,
            )
            layer_result: dict = {
                "reproduction": reproduction,
                "router_bias_equals_export": bool(torch.equal(bias, export_bias)),
                "lengths": list(layout.lengths),
                "piece_rows": args.pad_tokens or cuda_graph_padded_tokens(sum(layout.lengths)),
                "floor_piece_rows": cuda_graph_padded_tokens(sum(layout.lengths) + args.floor_extra_tokens),
            }
            flat_input = layout.flatten(layer_input)
            chained = run_vllm(
                replay,
                layout,
                shape,
                None,
                baseline.tensors,
                flat_input,
                config_tokens=args.moe_config_tokens,
                pad_tokens=args.pad_tokens,
            )
            extra = torch.randn(args.floor_extra_tokens, shape.hidden, generator=generator, device="cuda").to(
                flat_input.dtype
            )
            floor_tokens = None if args.moe_config_tokens is None else args.moe_config_tokens + args.floor_extra_tokens
            floor = run_vllm(
                replay,
                layout,
                shape,
                None,
                baseline.tensors,
                flat_input,
                config_tokens=floor_tokens,
                pad_tokens=None,
                extra_input=extra,
            )
            layer_result["floor"] = stats_json(compare_regions(chained, floor))
            sweeps = []
            for pinned in pinned_pieces:
                pinned_replay = LayerReplay(
                    pieces=pinned,
                    shape=shape,
                    weights=weights,
                    cos_sin=cos_sin,
                    layer=layer,
                    router_bias=bias,
                    ep_size=args.ep_size,
                    home_rank=args.home_rank,
                    order=order,
                )
                sweeps.append(
                    run_vllm(
                        pinned_replay,
                        layout,
                        shape,
                        None,
                        baseline.tensors,
                        flat_input,
                        config_tokens=args.moe_config_tokens,
                        pad_tokens=args.pad_tokens,
                    )
                )
            if len(sweeps) > 1:
                layer_result["config_floor"] = stats_json(compare_regions(sweeps[0], sweeps[-1]))
            variants = {"baseline": {}}
            for text in args.numerics:
                variants[text] = {flag: True for flag in text.split(",") if flag}
            for label, flags in variants.items():
                trainer_run = (
                    baseline
                    if not flags
                    else run_layer(layer_module, layer_input, rotary, next_norm=next_norm, numerics=flags)
                )
                trainer = trainer_region_tensors(trainer_run.tensors, layout, shape, trainer_run.next_norm)
                isolated = run_vllm(
                    replay,
                    layout,
                    shape,
                    trainer,
                    trainer_run.tensors,
                    flat_input,
                    config_tokens=args.moe_config_tokens,
                    pad_tokens=args.pad_tokens,
                )
                trainer_with_slots = {
                    **trainer,
                    "expert_slots": isolated.pop("trainer_expert_slots"),
                    "ep_sum": trainer["routed"],
                }
                ids, weights_used = trainer_routing(trainer_run.tensors, layout, bias, shape.top_k)
                vllm_ids, vllm_weights = isolated.pop("routing")
                chained_ids, chained_weights = chained.get("routing", (None, None))
                layer_result[label] = {
                    "isolated": stats_json(compare_regions(isolated, trainer_with_slots)),
                    "chained": stats_json(compare_regions(chained, trainer)),
                    "routing_isolated": routing_agreement(vllm_ids, vllm_weights, ids, weights_used),
                    "routing_chained": routing_agreement(chained_ids, chained_weights, ids, weights_used),
                    "flags": flags,
                }
            if layer == 0:
                layer_result["embedding"] = embedding_check(
                    replay, provider, weights, layout, sequences, attention_mask, flat_input
                )
            results["layers"][str(layer)] = layer_result
            del layer_module, replay
            gc.collect()
            torch.cuda.empty_cache()
        results["kernel_configs"] = chosen_configs(pieces, candidates)
    payload = json.dumps(results, indent=1, sort_keys=True, default=str)
    io.write_bytes_atomic(join_resource_path(args.output, "harness.json"), payload.encode())
    io.write_bytes_atomic(join_resource_path(args.output, "harness.md"), render_markdown(results).encode())
    print(render_markdown(results), flush=True)
    print("PASS mismatch harness", args.output, flush=True)


def embedding_check(replay, provider, weights, layout, sequences, attention_mask, flat_input) -> dict:
    """Layer 0's input: vLLM's compiled embedding path from token ids against the trainer's captured input."""
    token_ids = torch.cat(
        [sequences[index][attention_mask[index].bool()][:length] for index, length in enumerate(layout.lengths)]
    ).cuda()
    result, piece = replay.run_pre_attention(layout.requests, flat_input, token_ids, None, embedding_path=True)
    stored = {(value.launch.statement, value.variable): value.tensor for value in result.values}
    vllm = {role: stored[(ref.statement, ref.variable)] for role, ref in piece.values.items()}
    embed_norm = build_gated_norm(provider)
    load_hf_weights(embed_norm, "embed_norm.", weights)
    trainer = gated_norm_regions(embed_norm, weights["model.embed_tokens.weight"][token_ids])
    pairs = {
        "embed_rms": (vllm["embed_rms"], trainer["rms"]),
        "embed_gate_down": (vllm["embed_gate_down"], trainer["gate_down"]),
        "embed_gate_act": (vllm["embed_gate_act"], trainer["gate_act"]),
        "embed_gate_up": (vllm["embed_gate_up"], trainer["gate_up"]),
        "input_vllm_vs_trainer_module": (vllm["next.residual"], trainer["out"]),
        "input_vllm_vs_capture": (vllm["next.residual"], flat_input),
        "input_trainer_module_vs_capture": (trainer["out"], flat_input),
    }
    return {name: compare(a, b).to_json() for name, (a, b) in pairs.items()}


if __name__ == "__main__":
    main()
