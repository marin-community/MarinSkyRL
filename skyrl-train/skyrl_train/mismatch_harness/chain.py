"""The whole Snowball forward, token ids to log-probabilities, on compiled vLLM's archived kernels and on the trainer.

One GPU, layer by layer: each decoder layer's weights are loaded once, used by both sides, then freed.

- vLLM: the archived compiled pieces in order (the first piece from token ids, a middle piece per layer, the last
  piece), with FA3 and the emulated expert-parallel MoE between them, as one prefill step holding every request (a
  re-read data-parallel rank's step), then the LM head and ``log_softmax(dtype=float32)`` per request, as
  ``compute_logits`` and ``Sampler.compute_logprobs`` compute prompt log-probabilities. The pieces' autotuned
  reductions are pinned to chosen launch configs, so a run is reproducible.
- The trainer: the embedding lookup and gated norm, the Megatron Grug layers under each numerics variant (one
  sequence per forward, as the probe's micro-batches hold one), the final gated norm, the output layer's GEMM and the
  probe's log-probability function. It replays routes, by default the frozen re-read's (as ``reread_replay`` does).
  A variant with the pseudo-flag ``pp2`` drops the residual hand-off between layers 12 and 13, where the probe's
  two pipeline stages meet.

Reported per variant: per-layer byte equality of the residual stream against vLLM's, the final hidden state, the
logits and the log-probabilities against the vLLM replay, and the log-probabilities against the archived re-read and
against the probe's own trainer scores of the same rows.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from marinskyrl.resource_locator import join_resource_path
from megatron.core import parallel_state

from skyrl_train.distributed.megatron.model_utils import from_parallel_logits_to_logprobs, vllm_prompt_logprobs
from skyrl_train.io import io
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore
from skyrl_train.mismatch_harness.expert_parallel import ReduceOrder
from skyrl_train.mismatch_harness.harness import LayerReplay, attention_inputs, piece_regions
from skyrl_train.mismatch_harness.numerics import compare
from skyrl_train.mismatch_harness.pieces import PieceKind
from skyrl_train.mismatch_harness.regions import POST_ATTENTION_ROLES, PRE_ATTENTION_ROLES
from skyrl_train.mismatch_harness.run import (
    PieceConfigs,
    layer_weight_names,
    load_pinned_pieces,
    parse_variants,
    stage_config,
)
from skyrl_train.mismatch_harness.trainer_side import (
    build_gated_norm,
    build_layer,
    grug_provider,
    load_hf_weights,
    rotary_embedding,
    single_rank_megatron,
)
from skyrl_train.mismatch_harness.vllm_side import GrugShape, Requests, cos_sin_cache, vllm_config_context
from skyrl_train.mismatch_probe.numerics import grug_numerics
from skyrl_train.models.grug_megatron import NormRole, _take_hand_off, clear_numerics_handoffs
from skyrl_train.models.megatron_router_replay import LayerReplayHandle, MegatronRouterReplay, VllmExpertParallel

# The pseudo-flag of a variant that drops the residual hand-off where the probe's pipeline stages meet.
PP_BOUNDARY_FLAG = "pp2"
# The last decoder layer of the probe's first pipeline stage (PP 2, 13 layers per stage).
PP_BOUNDARY_LAYER = 12
GLOBAL_WEIGHTS = (
    "model.embed_tokens.weight",
    "model.embed_norm.weight",
    "model.embed_gated_norm.down_proj.weight",
    "model.embed_gated_norm.up_proj.weight",
    "model.norm.weight",
    "model.final_gated_norm.down_proj.weight",
    "model.final_gated_norm.up_proj.weight",
    "lm_head.weight",
)


@dataclass(frozen=True)
class Row:
    """One probe row: the full token sequence the re-read scored and what the archives recorded for it."""

    index: int
    sample_id: str
    tokens: list[int]
    prompt_length: int
    reread_logprobs: np.ndarray
    reread_again_logprobs: np.ndarray
    routes: np.ndarray
    """The re-read's routes ``[tokens - 1, layers, top_k]`` for every input position, vLLM's slot order."""
    trainer_scores: dict[str, np.ndarray]


def load_rows(uri: str) -> list[Row]:
    rows = []
    for record in pickle.loads(io.read_bytes(uri)):
        if record["prompt_token_ids"] != record["trainer_prompt_ids"]:
            raise ValueError(f"row {record['batch_position']}: vLLM and trainer prompts differ")
        if record["vllm_output_ids"] != record["trainer_input_ids"]:
            raise ValueError(f"row {record['batch_position']}: vLLM and trainer responses differ")
        tokens = record["prompt_token_ids"] + record["vllm_output_ids"]
        rows.append(
            Row(
                index=record["batch_position"],
                sample_id=record["sample_id"],
                tokens=tokens,
                prompt_length=len(record["prompt_token_ids"]),
                reread_logprobs=np.asarray(record["reread_logprobs"], np.float32),
                reread_again_logprobs=np.asarray(record["reread_again_logprobs"], np.float32),
                routes=record["reread_routes"],
                trainer_scores={label: np.asarray(values, np.float32) for label, values in record["trainer"].items()},
            )
        )
    return rows


class WeightStream:
    """Hugging Face tensors of one layer at a time, from the object-store export, on the GPU."""

    def __init__(self, model_uri: str, config_dir: Path):
        self.store = RemoteSafetensorsTensorStore(model_uri, config_dir)

    def load(self, names: list[str]) -> dict[str, torch.Tensor]:
        return {name: tensor.cuda() for name, tensor in self.store.load_tensors(sorted(set(names))).items()}


def logprob_stats(target: np.ndarray, reference: np.ndarray) -> dict:
    """Byte equality and |Δ| of two fp32 log-probability vectors, with the mean K3 of their ratio."""
    if target.shape != reference.shape:
        raise ValueError(f"log-probabilities of shape {target.shape} against {reference.shape}")
    delta = target.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(delta)
    return {
        "tokens": int(delta.size),
        "byte_equal_fraction": float((target.view(np.int32) == reference.view(np.int32)).mean()) if delta.size else 1.0,
        "abs_mean": float(absolute.mean()) if delta.size else 0.0,
        "abs_p99": float(np.quantile(absolute, 0.99)) if delta.size else 0.0,
        "abs_max": float(absolute.max()) if delta.size else 0.0,
        "k3_mean": float(np.mean(np.expm1(delta) - delta)) if delta.size else 0.0,
    }


def response_slice(row: Row) -> slice:
    """Positions of the per-position log-probabilities (``tokens - 1`` of them) that score response tokens."""
    return slice(row.prompt_length - 1, len(row.tokens) - 1)


@dataclass
class TrainerChain:
    """One trainer numerics variant's state: each sequence's residual stream ``[S, 1, H]``."""

    label: str
    flags: dict[str, bool]
    pp_boundary: bool
    hidden: list[torch.Tensor]
    layers: list[dict] | None = None


@contextmanager
def recorded_regions(layer) -> Iterator[dict[str, torch.Tensor]]:
    """The trainer layer's region tensors of one forward, by the harness's region names, ``[S, ...]`` for one sequence."""
    records: dict[str, torch.Tensor] = {}

    def keep(name: str, tensor: torch.Tensor) -> None:
        if name not in records:
            records[name] = tensor.detach().clone().reshape(tensor.shape[0], -1)

    def first(value):
        return value[0] if isinstance(value, (tuple, list)) else value

    def attention_inputs(module, args) -> None:
        for name, tensor in zip(("query", "key", "v_proj"), args[:3], strict=True):
            keep(name, tensor)

    handles = [
        layer.register_forward_pre_hook(
            lambda module, args, kwargs: keep("input", kwargs["hidden_states"]), with_kwargs=True
        ),
        layer.register_forward_hook(lambda module, args, output: keep("output", first(output))),
        layer.input_layernorm.down_proj.register_forward_pre_hook(lambda module, args: keep("attn_rms", args[0])),
        layer.input_layernorm.up_proj.register_forward_hook(lambda module, args, output: keep("attn_gate_up", output)),
        layer.input_layernorm.register_forward_hook(lambda module, args, output: keep("attention_norm", output)),
        layer.self_attention.q_layernorm.register_forward_pre_hook(lambda module, args: keep("q_proj", args[0])),
        layer.self_attention.k_layernorm.register_forward_pre_hook(lambda module, args: keep("k_proj", args[0])),
        layer.self_attention.core_attention.register_forward_pre_hook(attention_inputs),
        layer.self_attention.core_attention.register_forward_hook(
            lambda module, args, output: keep("core_attention", first(output))
        ),
        layer.self_attention.attn_gate.register_forward_hook(
            lambda module, args, output: keep("attn_gate", first(output))
        ),
        layer.self_attention.linear_proj.register_forward_pre_hook(lambda module, args: keep("xsa_gate", args[0])),
        layer.pre_mlp_layernorm.register_forward_pre_hook(
            lambda module, args: keep("residual_after_attention", args[0])
        ),
        layer.pre_mlp_layernorm.down_proj.register_forward_pre_hook(lambda module, args: keep("mlp_rms", args[0])),
        layer.pre_mlp_layernorm.register_forward_hook(lambda module, args, output: keep("mlp_norm", output)),
        layer.mlp.router.register_forward_hook(lambda module, args, output: keep("router_probs", output[0])),
        layer.mlp.shared_experts.linear_fc2.register_forward_pre_hook(lambda module, args: keep("shared_act", args[0])),
        layer.mlp.shared_experts.linear_fc2.register_forward_hook(
            lambda module, args, output: keep("shared_down", first(output))
        ),
    ]
    router = layer.mlp.router
    routing = router.routing
    dispatcher = layer.mlp.token_dispatcher
    combine = dispatcher.combine_postprocess

    def recorded_routing(logits, padding_mask=None):
        keep("router_logits", logits)
        return routing(logits, padding_mask)

    def recorded_combine(permuted):
        routed = combine(permuted)
        keep("routed", routed)
        return routed

    router.routing = recorded_routing
    dispatcher.combine_postprocess = recorded_combine
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()
        router.routing = routing
        dispatcher.combine_postprocess = combine


def region_differences(trainer: dict[str, torch.Tensor], vllm: dict[str, torch.Tensor]) -> list[dict]:
    """Per region in forward order: how many token rows differ, the first few differing positions, the largest ulp."""
    order = (
        "input",
        "attn_rms",
        "attn_gate_up",
        "attention_norm",
        "q_proj",
        "k_proj",
        "v_proj",
        "query",
        "key",
        "core_attention",
        "attn_gate",
        "xsa_gate",
        "residual_after_attention",
        "mlp_rms",
        "mlp_norm",
        "router_logits",
        "combine_weights",
        "expert_ids",
        "routed",
        "shared_act",
        "shared_down",
        "output",
    )
    found = []
    for name in order:
        if name not in trainer or name not in vllm:
            continue
        left, right = trainer[name], vllm[name].reshape(trainer[name].shape).to(trainer[name].dtype)
        if left.dtype.is_floating_point:
            view = {2: torch.int16, 4: torch.int32}[left.element_size()]
            same = left.contiguous().view(view) == right.contiguous().view(view)
        else:
            same = left == right
        rows = (~same).any(dim=-1).nonzero().flatten().tolist()
        entry = {"region": name, "rows_differing": len(rows), "first_positions": rows[:8]}
        if rows and left.dtype.is_floating_point:
            entry["elements_differing"] = int((~same).sum())
            entry["max_ulp"] = compare(left, right).max_ulp
        found.append(entry)
    return found


def run_trainer_layer(layer, hidden: torch.Tensor, rotary, flags: dict, targets: torch.Tensor | None) -> torch.Tensor:
    """The layer on one sequence ``[S, 1, H]`` under ``flags``; ``targets`` ``[S, top_k]`` replays routes (last row native)."""
    router = layer.mlp.router
    controller = None
    rows = hidden.shape[0]
    if targets is not None:
        controller = MegatronRouterReplay([0], recompute_enabled=False)
        router.router_replay = LayerReplayHandle(controller, 0)
        mask = torch.ones(rows, dtype=torch.bool, device=hidden.device)
        mask[-1] = False
        controller.begin_forward(
            {0: targets},
            mask,
            vllm_expert_parallel=VllmExpertParallel(torch.zeros(rows, dtype=torch.long, device=hidden.device), 8),
        )
    try:
        with torch.no_grad(), grug_numerics(**flags):
            output, _ = layer(hidden_states=hidden, attention_mask=None, rotary_pos_emb=rotary(rows))
        if controller is not None:
            controller.end_forward()
            controller.assert_drained()
    finally:
        router.router_replay = None
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", required=True, help="pickle of probe rows (p1_chain_rows.py)")
    parser.add_argument("--output-code", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, help="run only the first N layers (development)")
    parser.add_argument("--pad-tokens", type=int, default=8192, help="rows the compiled pieces run on")
    parser.add_argument("--moe-config-tokens", type=int, default=65536)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--home-rank", type=int, default=0)
    parser.add_argument("--norm-block", type=int, default=4096)
    parser.add_argument("--post-attention-norm-blocks", type=int, nargs="+", default=[4096])
    parser.add_argument("--qk-xblock", type=int, default=8)
    parser.add_argument("--route-source", choices=("reread", "vllm", "native"), default="reread")
    parser.add_argument("--numerics", action="append", default=[], help="[label=]flags (pseudo-flag pp2)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--localize",
        type=int,
        nargs=2,
        metavar=("LAYER", "ROW"),
        help="compare every region of LAYER for probe row ROW (batch position) between --localize-variant and vLLM",
    )
    parser.add_argument("--localize-variant", help="the variant label --localize compares")
    args = parser.parse_args()
    if (args.localize is None) != (args.localize_variant is None):
        parser.error("--localize and --localize-variant go together")

    results: dict = {"arguments": vars(args), "layers": [], "variants": {}}
    with tempfile.TemporaryDirectory() as directory, single_rank_megatron(args.seed), vllm_config_context():
        config_dir = Path(directory)
        config = stage_config(args.model, config_dir)
        shape = GrugShape.from_config(config)
        layers = args.layers or shape.layers
        rows = load_rows(args.rows)
        lengths = tuple(len(row.tokens) for row in rows)
        requests = Requests(lengths, padded=args.pad_tokens)
        tokens = requests.tokens
        offsets = np.cumsum((0, *lengths))
        results["rows"] = [{"index": row.index, "sample": row.sample_id, "tokens": len(row.tokens)} for row in rows]
        _, provider = grug_provider(str(config_dir))
        stream = WeightStream(args.model, config_dir)
        weights = stream.load(list(GLOBAL_WEIGHTS) + layer_weight_names(0, experts=False))
        rotary = rotary_embedding(provider).cuda()
        cos_sin = cos_sin_cache(shape)
        order = ReduceOrder.RING

        vllm_runs = {}
        for block in args.post_attention_norm_blocks:
            pieces, chosen = load_pinned_pieces(args.output_code, PieceConfigs(args.norm_block, block, args.qk_xblock))
            vllm_runs[block] = {"pieces": pieces, "result": None, "routes": [], "outputs": [], "configs": chosen}
        results["vllm_configs"] = {str(block): run["configs"] for block, run in vllm_runs.items()}
        reference_block = args.post_attention_norm_blocks[0]
        flat_ids = torch.tensor([token for row in rows for token in row.tokens], dtype=torch.int32, device="cuda")

        embed_norm = build_gated_norm(provider, NormRole.EMBEDDING)
        load_hf_weights(embed_norm, "embed_norm.", weights)
        variants = []
        # Hand-offs pass each variant's residual from one layer's output to the next layer's norm for the whole
        # forward; they are keyed by the tensors, so the variants' entries do not meet.
        clear_numerics_handoffs()
        for label, flags in parse_variants(args.numerics).items():
            numerics = {flag: value for flag, value in flags.items() if flag != PP_BOUNDARY_FLAG}
            chain = TrainerChain(label, numerics, bool(flags.get(PP_BOUNDARY_FLAG)), [], [])
            with torch.no_grad(), grug_numerics(**numerics):
                for row in rows:
                    ids = torch.tensor(row.tokens, dtype=torch.long, device="cuda")
                    embedded = weights["model.embed_tokens.weight"][ids].view(len(row.tokens), 1, -1)
                    chain.hidden.append(embed_norm(embedded))
            variants.append(chain)
        if args.localize_variant is not None and args.localize_variant not in {chain.label for chain in variants}:
            raise ValueError(f"--localize-variant {args.localize_variant!r} is not one of the variants")

        # The vLLM side's region tensors of the localized layer, every request's rows.
        vllm_regions: dict[str, torch.Tensor] = {}
        for layer in range(layers):
            prefix = f"model.layers.{layer}."
            names = layer_weight_names(layer, experts=True)
            if layer + 1 < shape.layers:
                names += layer_weight_names(layer + 1, experts=False)
            weights.update(stream.load([name for name in names if name not in weights]))
            layer_entry: dict = {"layer": layer}
            for block, run in vllm_runs.items():
                replay = LayerReplay(
                    pieces=run["pieces"],
                    shape=shape,
                    weights=weights,
                    cos_sin=cos_sin,
                    layer=layer,
                    router_bias=weights[prefix + "mlp.router.bias"].float(),
                    ep_size=args.ep_size,
                    home_rank=args.home_rank,
                    order=order,
                    fa3_step_tokens=tokens,
                )
                if run["result"] is None:
                    run["result"], run["piece"] = replay.run_pre_attention(
                        requests, torch.empty(0), flat_ids, None, embedding_path=True
                    )
                result = run["result"]
                localizing = args.localize is not None and args.localize[0] == layer and block == reference_block
                if localizing:
                    vllm_regions.update(piece_regions(result, run["piece"], PRE_ATTENTION_ROLES))
                inputs = attention_inputs(result, shape)
                outputs = [output for output in result.outputs if isinstance(output, torch.Tensor)]
                attn_in, residual = outputs[-2], outputs[-1]
                value = inputs["value"].reshape(requests.rows, -1)
                attention = replay.attention(inputs["query"], inputs["key"], inputs["value"], requests)
                moe = replay.moe(args.moe_config_tokens, None)
                run["result"], piece = replay.run_post_attention(
                    requests, attention, value, attn_in, residual, moe, None
                )
                run["piece"] = piece
                if localizing:
                    vllm_regions.update(
                        {
                            "query": inputs["query"],
                            "key": inputs["key"],
                            "core_attention": attention,
                            "combine_weights": moe.records[-1].vllm_weights,
                            "expert_ids": moe.records[-1].vllm_ids.long(),
                        }
                    )
                    post = piece_regions(run["result"], piece, POST_ATTENTION_ROLES)
                    post["attn_gate"] = post["attn_gate"][:, : shape.heads]
                    vllm_regions.update({name: tensor for name, tensor in post.items() if name not in vllm_regions})
                run["routes"].append(moe.records[-1].vllm_ids[:tokens].long())
                if piece.kind is not PieceKind.LAST:
                    run["outputs"].append(piece_regions(run["result"], piece, POST_ATTENTION_ROLES)["output"][:tokens])
                del moe, replay
            reference_run = vllm_runs[reference_block]
            vllm_routes = reference_run["routes"][-1]
            archived = [torch.from_numpy(row.routes[:, layer]).cuda() for row in rows]
            layer_entry["vllm_routes_vs_reread"] = float(
                torch.cat(
                    [
                        (vllm_routes[offsets[b] : offsets[b + 1] - 1] == archived[b]).all(dim=-1).float()
                        for b in range(len(rows))
                    ]
                )
                .mean()
                .item()
            )
            trainer_layer = build_layer(provider, layer)
            load_hf_weights(trainer_layer, f"decoder.layers.{layer}.", weights)
            for chain in variants:
                for b, row in enumerate(rows):
                    targets = None
                    if args.route_source != "native":
                        source = (
                            archived[b]
                            if args.route_source == "reread"
                            else vllm_routes[offsets[b] : offsets[b + 1] - 1]
                        )
                        targets = torch.cat((source, source[-1:]))
                    target = args.localize == [layer, row.index] and chain.label == args.localize_variant
                    with recorded_regions(trainer_layer) if target else nullcontext({}) as trainer_regions:
                        chain.hidden[b] = run_trainer_layer(
                            trainer_layer, chain.hidden[b], rotary, chain.flags, targets
                        )
                    if target:
                        rows_of_b = slice(int(offsets[b]), int(offsets[b + 1]))
                        ids = vllm_regions["expert_ids"][rows_of_b]
                        trainer_regions["expert_ids"] = targets[: ids.shape[0]].long()
                        trainer_regions["combine_weights"] = trainer_regions["router_probs"].float().gather(1, ids)
                        trainer_regions["attn_gate"] = trainer_regions["attn_gate"][:, : shape.heads]
                        found = region_differences(
                            trainer_regions, {name: tensor[rows_of_b] for name, tensor in vllm_regions.items()}
                        )
                        results["localize"] = {
                            "layer": layer,
                            "row": row.index,
                            "variant": chain.label,
                            "regions": found,
                        }
                        for entry in found:
                            print("LOCALIZE", json.dumps(entry), flush=True)
                    if chain.pp_boundary and layer == PP_BOUNDARY_LAYER:
                        _take_hand_off(chain.hidden[b])
                if layer + 1 < shape.layers:
                    trainer_output = torch.cat([hidden.view(-1, shape.hidden) for hidden in chain.hidden])
                    reference = reference_run["outputs"][-1]
                    stats = compare(reference, trainer_output)
                    rows_equal = (reference.view(torch.int16) == trainer_output.view(torch.int16)).all(dim=-1)
                    per_row = []
                    for b in range(len(rows)):
                        equal_rows = rows_equal[offsets[b] : offsets[b + 1]]
                        differing = (~equal_rows).nonzero().flatten()
                        per_row.append(
                            {
                                "token_rows_equal": float(equal_rows.float().mean().item()),
                                "first_differing_position": int(differing[0].item()) if differing.numel() else None,
                            }
                        )
                    chain.layers.append({"layer": layer, "output": stats.to_json(), "rows": per_row})
            del trainer_layer
            for name in [name for name in weights if name.startswith(prefix)]:
                del weights[name]
            gc.collect()
            torch.cuda.empty_cache()
            layer_entry["vllm_config_floor_output"] = (
                compare(*(run["outputs"][-1] for run in vllm_runs.values())).to_json()
                if len(vllm_runs) > 1 and layer + 1 < shape.layers
                else None
            )
            results["layers"].append(layer_entry)
            print(json.dumps(layer_entry), flush=True)
            for chain in variants:
                if chain.layers and chain.layers[-1]["layer"] == layer:
                    output = chain.layers[-1]["output"]
                    differing_rows = {
                        rows[b].index: (round(entry["token_rows_equal"], 4), entry["first_differing_position"])
                        for b, entry in enumerate(chain.layers[-1]["rows"])
                        if entry["first_differing_position"] is not None
                    }
                    print(
                        f"LAYER {layer} {chain.label}: output byte-equal {output['byte_equal_fraction']:.6f} "
                        f"max ulp {output['max_ulp']}; rows with a differing token (equal fraction, first position): "
                        f"{differing_rows if len(differing_rows) <= 6 else len(differing_rows)}",
                        flush=True,
                    )
        if layers < shape.layers:
            results["complete"] = False
            finish(results, args)
            return

        lm_head = weights["lm_head.weight"]
        vllm_logprobs = {}
        vllm_hidden = {}
        for block, run in vllm_runs.items():
            final_hidden = run["result"].outputs[0][:tokens]
            vllm_hidden[block] = final_hidden
            per_row = []
            for b, row in enumerate(rows):
                logits = F.linear(final_hidden[offsets[b] : offsets[b + 1] - 1], lm_head)
                next_ids = torch.tensor(row.tokens[1:], dtype=torch.long, device="cuda")
                values = logits.log_softmax(dim=-1, dtype=torch.float32).gather(-1, next_ids[:, None]).squeeze(-1)
                per_row.append((logits, values.cpu().numpy()))
            vllm_logprobs[block] = per_row
        results["vllm"] = {}
        for block, per_row in vllm_logprobs.items():
            response = np.concatenate([values[response_slice(row)] for row, (_, values) in zip(rows, per_row)])
            results["vllm"][str(block)] = {
                "vs_reread": logprob_stats(response, np.concatenate([row.reread_logprobs for row in rows])),
                "vs_reread_again": logprob_stats(response, np.concatenate([row.reread_again_logprobs for row in rows])),
                "per_row_vs_reread_byte_equal": [
                    float((values[response_slice(row)].view(np.int32) == row.reread_logprobs.view(np.int32)).mean())
                    for row, (_, values) in zip(rows, per_row)
                ],
            }
            if block != reference_block:
                results["vllm"][str(block)]["vs_reference_config"] = logprob_stats(
                    np.concatenate([values for _, values in per_row]),
                    np.concatenate([values for _, values in vllm_logprobs[reference_block]]),
                )

        final_norm = build_gated_norm(provider, NormRole.FINAL)
        load_hf_weights(final_norm, "decoder.final_layernorm.", weights)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        reference_rows = vllm_logprobs[reference_block]
        for chain in variants:
            hidden_final, logits_equal, own, trainer_softmax, vllm_softmax = [], [], [], [], []
            with torch.no_grad(), grug_numerics(**chain.flags):
                for b, row in enumerate(rows):
                    final = final_norm(chain.hidden[b])
                    hidden_final.append(final.view(-1, shape.hidden))
                    logits = torch.matmul(final, lm_head.t()).transpose(0, 1).contiguous()
                    ids = torch.tensor(row.tokens, dtype=torch.long, device="cuda")[None]
                    trainer_values = from_parallel_logits_to_logprobs(
                        logits, ids, 0, logits.shape[-1], tp_group, inference_only=True
                    )[0]
                    vllm_values = vllm_prompt_logprobs(logits, ids, None)[0]
                    trainer_softmax.append(trainer_values.cpu().numpy())
                    vllm_softmax.append(vllm_values.cpu().numpy())
                    own.append(vllm_softmax[-1] if chain.flags.get("vllm_log_softmax") else trainer_softmax[-1])
                    logits_equal.append(compare(reference_rows[b][0], logits[0, :-1]).to_json())
            all_reference = np.concatenate([values for _, values in reference_rows])
            response_reference = np.concatenate(
                [values[response_slice(row)] for row, (_, values) in zip(rows, reference_rows)]
            )
            reread = np.concatenate([row.reread_logprobs for row in rows])

            def responses(per_row: list[np.ndarray]) -> np.ndarray:
                return np.concatenate([values[response_slice(row)] for row, values in zip(rows, per_row)])

            entry = {
                "flags": chain.flags,
                "pp_boundary": chain.pp_boundary,
                "layers": chain.layers,
                "final_hidden": compare(vllm_hidden[reference_block], torch.cat(hidden_final)).to_json(),
                "logits_byte_equal_fraction": float(
                    np.average(
                        [s["byte_equal_fraction"] for s in logits_equal], weights=[s["elements"] for s in logits_equal]
                    )
                ),
                "logits_rows_all_equal_fraction": float(
                    np.average(
                        [s["rows_all_equal_fraction"] for s in logits_equal], weights=[len(r.tokens) - 1 for r in rows]
                    )
                ),
                "logprobs_vs_vllm_all_positions": logprob_stats(np.concatenate(own), all_reference),
                "logprobs_vs_vllm_response": logprob_stats(responses(own), response_reference),
                "trainer_log_softmax_vs_vllm_response": logprob_stats(responses(trainer_softmax), response_reference),
                "vllm_log_softmax_vs_vllm_response": logprob_stats(responses(vllm_softmax), response_reference),
                "logprobs_vs_reread_response": logprob_stats(responses(own), reread),
                "vs_probe_trainer_scores": {
                    label: logprob_stats(responses(own), np.concatenate([row.trainer_scores[label] for row in rows]))
                    for label in rows[0].trainer_scores
                },
            }
            results["variants"][chain.label] = entry
            print(
                f"VARIANT {chain.label}: logprobs byte-equal to vLLM replay "
                f"{entry['logprobs_vs_vllm_response']['byte_equal_fraction']:.4f} (all positions "
                f"{entry['logprobs_vs_vllm_all_positions']['byte_equal_fraction']:.4f}), p99 "
                f"{entry['logprobs_vs_vllm_response']['abs_p99']:.4g}; vs archived re-read byte-equal "
                f"{entry['logprobs_vs_reread_response']['byte_equal_fraction']:.4f} p99 "
                f"{entry['logprobs_vs_reread_response']['abs_p99']:.4g}",
                flush=True,
            )
        results["complete"] = True
    finish(results, args)


def finish(results: dict, args) -> None:
    payload = json.dumps(results, indent=1, sort_keys=True, default=_json_default)
    io.write_bytes_atomic(join_resource_path(args.output, "chain.json"), payload.encode())
    print(
        "CHAIN_RESULT " + json.dumps({k: v for k, v in results.items() if k != "layers"}, default=_json_default),
        flush=True,
    )
    print("PASS mismatch harness chain", args.output, flush=True)


def _json_default(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


if __name__ == "__main__":
    main()
