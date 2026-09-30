"""Validate the replay against compiled vLLM itself, on a truncated real-weights Grug export.

The export keeps the first ``N`` decoder layers of a Grug checkpoint (with ``N = 4`` the last layer is a
full-attention layer, as layer 3 of the full model is). Compiled vLLM serves it with TP1 on
``--dp-size`` GPUs: one engine with EP off, or one engine per data-parallel rank with expert
parallelism across them, as the probe's engines run. Each rank scores one prompt (the capture's
first sequence on rank 0, a shorter reversed copy on the others); the run keeps each rank's routes,
prompt logits and the Inductor output code it compiled. The harness then replays each rank's output
code on its tokens (with vLLM's FA3 and Triton fused-MoE kernels between the pieces, the MoE emulated
over the ranks with the step sizes the engine recorded) and requires byte-identical routes and logits.
The pieces are also text-diffed against a full-model probe's archived output code.

The engine's worker also records what reaches each FA3 call on the prompt's step, its rotary table and
the launch config each Inductor kernel ran with (``engine_taps``). The replay's own attention inputs
and outputs are compared with them layer by layer, and FA3 is rerun on the engine's own inputs, so a
failed validation names the first op whose inputs or outputs differ.

Example (two H100s, EP=2)::

    python -m skyrl_train.mismatch_harness.validation --model s3://.../hf-bf16-vllm --layers 4 --dp-size 2 \\
        --capture s3://.../pp-0.pt --full-output-code s3://.../engine-0/dp-0-ep-0 --output s3://.../validation
"""

from __future__ import annotations

import argparse
import difflib
import io as stdlib_io
import json
import multiprocessing
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from marinskyrl.resource_locator import join_resource_path, relative_resource_path
from safetensors.torch import load_file, save_file
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.utils.network_utils import get_open_port

from skyrl_train.io import io
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore
from skyrl_train.mismatch_harness import engine_taps
from skyrl_train.mismatch_harness.expert_parallel import ReduceOrder
from skyrl_train.mismatch_harness.harness import LayerReplay, attention_inputs
from skyrl_train.mismatch_harness.numerics import compare
from skyrl_train.mismatch_harness.output_code import is_graph_module, parse_output_code
from skyrl_train.mismatch_harness.pieces import classify
from skyrl_train.mismatch_harness.replay import load_module
from skyrl_train.mismatch_harness.vllm_side import (
    GrugShape,
    Requests,
    cos_sin_cache,
    fa3_num_splits,
    flash_attention_prefill,
    paged_kv_cache,
    vllm_config_context,
)

WEIGHT_INDEX = "model.safetensors.index.json"
PROMPT_LOGPROBS = 20
# Engine settings of the probe's vLLM that decide what it compiles and how it computes.
ENGINE_SETTINGS = {
    "dtype": "bfloat16",
    "enforce_eager": False,
    "tensor_parallel_size": 1,
    "enable_expert_parallel": False,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 1024,
    "moe_backend": "triton",
    "attention_backend": "FLASH_ATTN",
    "enable_return_routed_experts": True,
    "logprobs_mode": "raw_logits",
    "max_logprobs": PROMPT_LOGPROBS,
    "gpu_memory_utilization": 0.5,
    "trust_remote_code": True,
}


def truncated_export(source_uri: str, num_layers: int, destination: Path) -> None:
    """Write a Grug export that keeps the first ``num_layers`` decoder layers to ``destination / "model"``."""
    metadata = destination / "metadata"
    metadata.mkdir(parents=True)
    for path, _ in io.find_files(source_uri).items():
        relative = relative_resource_path(source_uri, path)
        if relative.endswith(".safetensors") or "/" in relative:
            continue
        (metadata / relative).write_bytes(io.read_bytes(join_resource_path(source_uri, relative)))
    index = json.loads((metadata / WEIGHT_INDEX).read_text())
    keep = [
        name
        for name in index["weight_map"]
        if not name.startswith("model.layers.") or int(name.split(".")[2]) < num_layers
    ]
    store = RemoteSafetensorsTensorStore(source_uri, metadata)
    model_dir = destination / "model"
    model_dir.mkdir()
    groups = [[name for name in keep if not name.startswith("model.layers.")]]
    groups += [[name for name in keep if name.startswith(f"model.layers.{layer}.")] for layer in range(num_layers)]
    weight_map = {}
    for number, names in enumerate(groups):
        shard = f"model-{number + 1:05d}-of-{len(groups):05d}.safetensors"
        loaded = store.load_tensors(names)
        save_file({name: tensor.contiguous() for name, tensor in loaded.items()}, model_dir / shard)
        weight_map.update({name: shard for name in names})
        del loaded
    for path in metadata.iterdir():
        if path.name not in (WEIGHT_INDEX, "config.json"):
            shutil.copy(path, model_dir / path.name)
    config = json.loads((metadata / "config.json").read_text())
    for key in ("num_hidden_layers", "num_layers"):
        if key in config:
            config[key] = num_layers
    (model_dir / "config.json").write_text(json.dumps(config, indent=1))
    (model_dir / WEIGHT_INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))


def export_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """Every tensor of a local safetensors export, on the GPU."""
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        tensors |= {name: tensor.cuda() for name, tensor in load_file(shard).items()}
    return tensors


# Rank r > 0 scores the capture's prompt reversed and cut by this many tokens per rank, so the ranks'
# steps differ in content and length.
OTHER_RANK_SHORTENING = 100


def rank_prompts(token_ids: list[int], dp_size: int) -> list[list[int]]:
    """One prompt per data-parallel rank: the capture's prompt, then shorter reversed copies."""
    return [token_ids] + [
        token_ids[::-1][: len(token_ids) - OTHER_RANK_SHORTENING * rank] for rank in range(1, dp_size)
    ]


def serve_rank(
    model_dir: str, token_ids: list[int], rank: int, dp_size: int, max_model_len: int, port: int, barrier, work: str
) -> None:
    """One compiled vLLM engine (data-parallel rank ``rank`` of ``dp_size``) scores one prompt.

    Runs in its own process. It saves the prompt's routes, prompt logits, the engine's taps and the
    output code it compiled into its own Inductor cache to ``work/served-rank{rank}.pt``.
    """
    work_dir = Path(work)
    inductor = work_dir / f"inductor-rank{rank}"
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    # Each rank compiles and autotunes into its own cache, so the replay reads that rank's choices.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)
    if dp_size > 1:
        # vLLM's offline data parallelism: one process per rank, coordinated through these variables.
        os.environ.update(
            {
                "VLLM_DP_RANK": str(rank),
                "VLLM_DP_RANK_LOCAL": str(rank),
                "VLLM_DP_SIZE": str(dp_size),
                "VLLM_DP_MASTER_IP": "127.0.0.1",
                "VLLM_DP_MASTER_PORT": str(port),
            }
        )
    settings = ENGINE_SETTINGS | {"enable_expert_parallel": dp_size > 1}
    llm = LLM(model=model_dir, max_model_len=max_model_len, worker_extension_cls=engine_taps.EXTENSION, **settings)
    llm.collective_rpc("tap_attention", args=(len(token_ids),))
    params = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=PROMPT_LOGPROBS, seed=0)
    barrier.wait()
    output = llm.generate([TokensPrompt(prompt_token_ids=token_ids)], params)[0]
    taps = work_dir / f"taps-rank{rank}.pt"
    llm.collective_rpc("save_taps", args=(str(taps), len(token_ids)))
    # No rank shuts its engine down while another still needs it for the MoE collectives.
    barrier.wait()
    logits = {
        position: {int(token): float(value.logprob) for token, value in entry.items()}
        for position, entry in enumerate(output.prompt_logprobs)
        if entry is not None
    }
    code = {
        str(path.relative_to(inductor)): path.read_text()
        for pattern in ("*.py", "*.best_config")
        for path in sorted(inductor.rglob(pattern))
    }
    served = {
        "rank": rank,
        "token_ids": token_ids,
        "routes": np.asarray(output.outputs[0].routed_experts),
        "logits": logits,
        "code": code,
        "taps": torch.load(taps, weights_only=False),
    }
    torch.save(served, work_dir / f"served-rank{rank}.pt")


def run_processes(target, arguments: list[tuple], *, together: bool) -> None:
    """Run ``target`` once per argument tuple, each in a spawned process; all at once or one by one."""
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=target, args=args) for args in arguments]
    batches = [processes] if together else [[process] for process in processes]
    for batch in batches:
        for process in batch:
            process.start()
        for process in batch:
            process.join()
    failed = {index: process.exitcode for index, process in enumerate(processes) if process.exitcode != 0}
    if failed:
        raise RuntimeError(f"{target.__name__} processes exited with codes {failed}")


def serve(prompts: list[list[int]], work: Path) -> None:
    """Serve ``len(prompts)`` data-parallel ranks (EP across them when more than one), one prompt each."""
    barrier = multiprocessing.get_context("spawn").Barrier(len(prompts))
    port = get_open_port()
    max_model_len = max(len(prompt) for prompt in prompts) + 16
    arguments = [
        (str(work / "model"), prompt, rank, len(prompts), max_model_len, port, barrier, str(work))
        for rank, prompt in enumerate(prompts)
    ]
    run_processes(serve_rank, arguments, together=True)


def replay_rank(work: str, rank: int, dp_size: int, archive: str) -> None:
    """Replay one rank's prompt in the Inductor cache its engine compiled into, as that engine ran it.

    Runs in its own process, so the replay's kernels load beside the rank's own ``.best_config`` files.
    Saves the comparison to ``work/result-rank{rank}.json``.
    """
    work_dir = Path(work)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(work_dir / f"inductor-rank{rank}")
    served = torch.load(work_dir / f"served-rank{rank}.pt", weights_only=False)
    shape = GrugShape.from_config(json.loads((work_dir / "model" / "config.json").read_text()))
    weights = export_tensors(work_dir / "model")
    with vllm_config_context():
        result = rank_result(served, shape, weights, dp_size, archive)
    (work_dir / f"result-rank{rank}.json").write_text(json.dumps(result, default=str))


@dataclass(frozen=True)
class ReplayTrace:
    """What the replay computed: routes ``[P, L, K]``, bf16 logits, each layer's FA3 call, kernel configs."""

    routes: torch.Tensor
    logits: torch.Tensor
    attention: list[dict[str, torch.Tensor]]
    kernel_configs: dict[str, list[str]]


def replay_model(
    code: dict[str, str],
    shape: GrugShape,
    weights: dict[str, torch.Tensor],
    token_ids: list[int],
    *,
    rows: int,
    ep_size: int,
    home_rank: int,
    moe_tokens: int,
) -> ReplayTrace:
    """Run the served model's own compiled pieces end to end on one rank's prompt.

    ``rows`` is the step's padded token count on this rank and ``moe_tokens`` the count the MoE ran on
    (every rank's padded tokens, gathered), both as the engine's taps recorded them.
    """
    pieces = {}
    for relative, text in code.items():
        if relative.endswith(".py") and is_graph_module(text):
            piece = classify(parse_output_code(text))
            pieces[(piece.kind, piece.rope)] = (piece, load_module(piece.code, f"validation_{relative}").__dict__)
    cos_sin = cos_sin_cache(shape)
    tokens = len(token_ids)
    requests = Requests((tokens,), padded=rows)
    ids = torch.tensor(token_ids, dtype=torch.int32, device="cuda")
    routes = []
    attention_calls = []
    result = None
    for layer in range(shape.layers):
        replay = LayerReplay(
            pieces=pieces,
            shape=shape,
            weights=weights,
            cos_sin=cos_sin,
            layer=layer,
            router_bias=weights[f"model.layers.{layer}.mlp.router.bias"].float(),
            ep_size=ep_size,
            home_rank=home_rank,
            order=ReduceOrder.RING,
        )
        if result is None:
            result, _ = replay.run_pre_attention(requests, torch.empty(0), ids, None, embedding_path=True)
        inputs = attention_inputs(result, shape)
        outputs = [output for output in result.outputs if isinstance(output, torch.Tensor)]
        attn_in, residual = outputs[-2], outputs[-1]
        value = inputs["value"].reshape(requests.rows, -1)
        attention = replay.attention(inputs["query"], inputs["key"], inputs["value"], requests)
        attention_calls.append(
            {name: inputs[name][:tokens].clone() for name in ("query", "key", "value")}
            | {"output": attention[:tokens].clone()}
        )
        moe = replay.moe(moe_tokens, None)
        result, _ = replay.run_post_attention(requests, attention, value, attn_in, residual, moe, None)
        routes.append(moe.records[-1].vllm_ids)
    final_hidden = result.outputs[0]
    logits = F.linear(final_hidden[: tokens - 1], weights["lm_head.weight"])
    kernels = [value for _, namespace in pieces.values() for value in namespace.values()]
    return ReplayTrace(
        routes=torch.stack(routes, dim=1)[:tokens],
        logits=logits,
        attention=attention_calls,
        kernel_configs=engine_taps.inductor_kernel_configs(kernels),
    )


def _layer_index(layer_name: str) -> int:
    return int(re.search(r"layers\.(\d+)\.", layer_name).group(1))


def _agreement(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Byte-equal fraction, max ulp and the first token row holding a differing element."""
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1).to(left.device)
    stats = compare(left, right)
    rows = (left != right).any(-1).nonzero().flatten()
    return {
        "byte_equal_fraction": stats.byte_equal_fraction,
        "max_ulp": stats.max_ulp,
        "first_differing_row": int(rows[0]) if rows.numel() else None,
    }


ENGINE_CALL_FIELDS = (
    "query_rows",
    "key_cache_stride",
    "block_size",
    "sliding_window",
    "scale",
    "softcap",
    "fa_version",
    "max_query_len",
    "max_seq_len",
    "seq_lens",
    "scheduler_metadata",
    "max_num_splits",
    "causal",
    "use_cascade",
)


def attention_diagnostics(engine: dict[str, Any], trace: ReplayTrace, shape: GrugShape) -> dict[str, Any]:
    """Compare the replay's FA3 inputs and outputs with the engine's; rerun FA3 on the engine's inputs."""
    by_layer = []
    for call in sorted(engine["attention"], key=lambda call: _layer_index(call["layer_name"])):
        layer = _layer_index(call["layer_name"])
        query, key, value = (call[name].cuda() for name in ("query", "key", "value"))
        requests = Requests(tuple(call["seq_lens"]))
        window = None if shape.is_long(layer) else shape.sliding_window
        splits = sorted({call["max_num_splits"], fa3_num_splits(requests.tokens), 0})
        rerun = {
            f"num_splits={count}": _agreement(
                flash_attention_prefill(
                    query, key, value, requests, window=window, scale=shape.head_dim**-0.5, num_splits=count
                ),
                call["output"],
            )
            for count in splits
        }
        key_cache, _, _ = paged_kv_cache(key, value, requests.lengths, call["block_size"])
        by_layer.append(
            {
                "layer": layer,
                "engine_call": {name: call[name] for name in ENGINE_CALL_FIELDS},
                "harness_call": {
                    "key_cache_stride": list(key_cache.stride()),
                    "window_size": [window - 1, 0] if window is not None else [-1, -1],
                    "num_splits": fa3_num_splits(requests.tokens),
                    "scale": shape.head_dim**-0.5,
                },
                "replay_vs_engine": {
                    name: _agreement(trace.attention[layer][name], call[name])
                    for name in ("query", "key", "value", "output")
                },
                "fa3_on_engine_inputs_vs_engine_output": rerun,
            }
        )
    return {"calls_recorded": len(engine["attention"]), "by_layer": by_layer}


def rotary_diagnostics(engine: dict[str, Any], shape: GrugShape, positions: int) -> dict[str, Any]:
    ours = cos_sin_cache(shape)[:positions].cpu()
    return {"engine_tables_equal_to_harness": [bool(torch.equal(table, ours)) for table in engine["cos_sin_cache"]]}


def kernel_config_diagnostics(engine: dict[str, list[str]], replay: dict[str, list[str]]) -> dict[str, Any]:
    shared = sorted(set(engine) & set(replay))
    return {
        "replay_kernels": len(replay),
        "replay_kernels_found_in_engine": len(shared),
        "differing": {
            name: {"engine": engine[name], "replay": replay[name]} for name in shared if engine[name] != replay[name]
        },
    }


def logits_diagnostics(served: dict[int, dict[int, float]], logits: torch.Tensor) -> dict[str, Any]:
    rows = logits.float().cpu()
    equal = {
        position: all(rows[position - 1, token].item() == value for token, value in values.items())
        for position, values in served.items()
    }
    differing = sorted(position for position, same in equal.items() if not same)
    return {
        "positions": len(equal),
        "positions_all_equal_fraction": sum(equal.values()) / max(len(equal), 1),
        "first_differing_position": differing[0] if differing else None,
    }


def diff_pieces(served: dict[str, str], archived_uri: str) -> dict[str, str]:
    """Text-diff each served piece against the archived full-model piece of the same kind."""
    archived = {}
    for path in io.find_files(archived_uri):
        relative = relative_resource_path(archived_uri, path)
        if not relative.endswith(".py"):
            continue
        text = io.read_bytes(join_resource_path(archived_uri, relative)).decode()
        if is_graph_module(text):
            piece = classify(parse_output_code(text))
            archived[f"{piece.kind}{'+rope' if piece.rope else ''}"] = text
    result = {}
    for relative, text in served.items():
        if not (relative.endswith(".py") and is_graph_module(text)):
            continue
        piece = classify(parse_output_code(text))
        key = f"{piece.kind}{'+rope' if piece.rope else ''}"
        reference = archived.get(key)
        if reference is None:
            result[key] = "no archived piece of this kind"
            continue
        changed = [
            line
            for line in difflib.unified_diff(reference.splitlines(), text.splitlines(), lineterm="", n=0)
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]
        result[key] = "identical" if not changed else f"{len(changed)} changed lines: {changed[:6]}"
    return result


def rank_result(served: dict, shape: GrugShape, weights: dict[str, torch.Tensor], dp_size: int, archive: str) -> dict:
    """Replay one rank's prompt with the step sizes its engine recorded and compare routes and logits."""
    token_ids = served["token_ids"]
    first = min(served["taps"]["attention"], key=lambda call: _layer_index(call["layer_name"]))
    rows = first["query_rows"]
    across = first["num_tokens_across_dp"]
    moe_tokens = rows if across is None else sum(across)
    trace = replay_model(
        served["code"],
        shape,
        weights,
        token_ids,
        rows=rows,
        ep_size=dp_size,
        home_rank=served["rank"],
        moe_tokens=moe_tokens,
    )
    diagnostics = {
        "attention": attention_diagnostics(served["taps"], trace, shape),
        "rotary_table": rotary_diagnostics(served["taps"], shape, len(token_ids)),
        "kernel_configs": kernel_config_diagnostics(served["taps"]["kernel_configs"], trace.kernel_configs),
        "logits": logits_diagnostics(served["logits"], trace.logits),
    }
    served_routes = torch.as_tensor(served["routes"]).to(trace.routes.device)
    replayed = trace.routes[: served_routes.shape[0]].to(served_routes.dtype)
    compared = equal = 0
    for position, values in served["logits"].items():
        for token, value in values.items():
            compared += 1
            equal += trace.logits[position - 1, token].float().item() == value
    same_order = replayed == served_routes
    same_set = torch.sort(replayed, dim=-1).values == torch.sort(served_routes, dim=-1).values
    first_differing = [
        int(position[0]) if (position := (~same_order[:, layer].all(-1)).nonzero().flatten()).numel() else None
        for layer in range(served_routes.shape[1])
    ]
    return {
        "rank": served["rank"],
        "prompt_tokens": len(token_ids),
        "piece_rows": rows,
        "num_tokens_across_dp": across,
        "moe_tokens": moe_tokens,
        "routes_shape": list(served_routes.shape),
        "routes_equal_fraction": same_order.float().mean().item(),
        "routes_rows_all_equal": bool(torch.equal(replayed, served_routes)),
        "routes_equal_fraction_by_layer": same_order.float().mean(dim=(0, 2)).tolist(),
        "expert_sets_equal_fraction_by_layer": same_set.all(-1).float().mean(dim=0).tolist(),
        "first_differing_position_by_layer": first_differing,
        "logits_compared": compared,
        "logits_equal_fraction": equal / max(compared, 1),
        "piece_diff_against_full_model": diff_pieces(served["code"], archive),
        "best_configs": sum(1 for relative in served["code"] if relative.endswith(".best_config")),
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dp-size", type=int, default=1, help="data-parallel ranks, one GPU each; EP when above 1")
    parser.add_argument("--capture", required=True, help="a trainer capture whose first sequence is the prompt")
    parser.add_argument("--full-output-code", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    capture = torch.load(stdlib_io.BytesIO(io.read_bytes(args.capture)), map_location="cpu", weights_only=False)
    mask = capture["attention_mask"][0].bool()
    prompts = rank_prompts(capture["sequences"][0][mask].tolist(), args.dp_size)
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        truncated_export(args.model, args.layers, work)
        serve(prompts, work)
        arguments = [(str(work), rank, args.dp_size, args.full_output_code) for rank in range(args.dp_size)]
        run_processes(replay_rank, arguments, together=False)
        ranks = [json.loads((work / f"result-rank{rank}.json").read_text()) for rank in range(args.dp_size)]
        codes = {
            rank: torch.load(work / f"served-rank{rank}.pt", weights_only=False)["code"] for rank in range(args.dp_size)
        }
    result = {
        "dp_size": args.dp_size,
        "engine_settings": ENGINE_SETTINGS | {"enable_expert_parallel": args.dp_size > 1},
        "ranks": ranks,
    }
    payload = json.dumps(result, indent=1, sort_keys=True, default=str)
    io.write_bytes_atomic(join_resource_path(args.output, "validation.json"), payload.encode())
    for rank, code in codes.items():
        for relative, text in code.items():
            io.write_bytes_atomic(
                join_resource_path(args.output, "output-code", f"rank{rank}", relative), text.encode()
            )
    print(payload, flush=True)
    passed = all(rank["routes_rows_all_equal"] and rank["logits_equal_fraction"] == 1.0 for rank in ranks)
    print(("PASS" if passed else "FAIL") + " replay validation", args.output, flush=True)


if __name__ == "__main__":
    main()
