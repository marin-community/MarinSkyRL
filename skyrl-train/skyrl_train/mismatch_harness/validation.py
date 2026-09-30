"""Validate the replay against compiled vLLM itself, on a truncated real-weights Grug export.

The export keeps the first ``N`` decoder layers of a Grug checkpoint (with ``N = 4`` the last layer is a
full-attention layer, as layer 3 of the full model is). Compiled vLLM serves it on one GPU (TP1, EP
off) and scores one prompt; the run keeps its routes, its prompt logits and the Inductor output code it
compiled. The harness then replays that output code on the same tokens (with vLLM's FA3 and Triton
fused-MoE kernels between the pieces) and requires byte-identical routes and logits. The pieces are
also text-diffed against a full-model probe's archived output code.

Example (one H100)::

    python -m skyrl_train.mismatch_harness.validation --model s3://.../hf-bf16-vllm --layers 4 \\
        --capture s3://.../pp-0.pt --full-output-code s3://.../engine-0/dp-0-ep-0 --output s3://.../validation
"""

from __future__ import annotations

import argparse
import difflib
import gc
import io as stdlib_io
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from marinskyrl.resource_locator import join_resource_path, relative_resource_path
from safetensors.torch import save_file
from torch._inductor.runtime.cache_dir_utils import cache_dir
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from skyrl_train.io import io
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore
from skyrl_train.mismatch_harness.expert_parallel import ReduceOrder
from skyrl_train.mismatch_harness.harness import LayerReplay, attention_inputs
from skyrl_train.mismatch_harness.output_code import is_graph_module, parse_output_code
from skyrl_train.mismatch_harness.pieces import classify
from skyrl_train.mismatch_harness.replay import load_module
from skyrl_train.mismatch_harness.vllm_side import GrugShape, Requests, cos_sin_cache, vllm_config_context

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


def truncated_export(source_uri: str, num_layers: int, destination: Path) -> dict[str, torch.Tensor]:
    """Write a Grug export that keeps the first ``num_layers`` decoder layers; return its tensors."""
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
    tensors: dict[str, torch.Tensor] = {}
    groups = [[name for name in keep if not name.startswith("model.layers.")]]
    groups += [[name for name in keep if name.startswith(f"model.layers.{layer}.")] for layer in range(num_layers)]
    weight_map = {}
    for number, names in enumerate(groups):
        shard = f"model-{number + 1:05d}-of-{len(groups):05d}.safetensors"
        loaded = store.load_tensors(names)
        save_file({name: tensor.contiguous() for name, tensor in loaded.items()}, model_dir / shard)
        weight_map.update({name: shard for name in names})
        tensors.update({name: tensor.cuda() for name, tensor in loaded.items()})
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
    return tensors


def serve(model_dir: Path, token_ids: list[int]) -> dict:
    """Score one prompt with compiled vLLM; return its routes, prompt logits and output code."""
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    llm = LLM(model=str(model_dir), max_model_len=len(token_ids) + 16, **ENGINE_SETTINGS)
    params = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=PROMPT_LOGPROBS, seed=0)
    output = llm.generate([TokensPrompt(prompt_token_ids=token_ids)], params)[0]
    completion = output.outputs[0]
    routes = np.asarray(completion.routed_experts)
    logits = {}
    for position, entry in enumerate(output.prompt_logprobs):
        if entry is None:
            continue
        logits[position] = {int(token): float(value.logprob) for token, value in entry.items()}
    root = Path(cache_dir())
    code = {
        str(path.relative_to(root)): path.read_text()
        for pattern in ("*.py", "*.best_config")
        for path in sorted(root.rglob(pattern))
    }
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return {"routes": routes, "logits": logits, "code": code}


def replay_model(code: dict[str, str], shape: GrugShape, weights: dict[str, torch.Tensor], token_ids: list[int]):
    """Run the served model's own compiled pieces end to end; return routes ``[P, L, K]`` and bf16 logits."""
    pieces = {}
    for relative, text in code.items():
        if relative.endswith(".py") and is_graph_module(text):
            piece = classify(parse_output_code(text))
            pieces[(piece.kind, piece.rope)] = (piece, load_module(piece.code, f"validation_{relative}").__dict__)
    cos_sin = cos_sin_cache(shape)
    requests = Requests((len(token_ids),))
    ids = torch.tensor(token_ids, dtype=torch.int32, device="cuda")
    routes = []
    result = None
    for layer in range(shape.layers):
        replay = LayerReplay(
            pieces=pieces,
            shape=shape,
            weights=weights,
            cos_sin=cos_sin,
            layer=layer,
            router_bias=weights[f"model.layers.{layer}.mlp.router.bias"].float(),
            ep_size=1,
            home_rank=0,
            order=ReduceOrder.RANK,
        )
        if result is None:
            result, _ = replay.run_pre_attention(requests, torch.empty(0), ids, None, embedding_path=True)
        inputs = attention_inputs(result, shape)
        outputs = [output for output in result.outputs if isinstance(output, torch.Tensor)]
        attn_in, residual = outputs[-2], outputs[-1]
        value = inputs["value"].reshape(requests.tokens, -1)
        attention = replay.attention(inputs["query"], inputs["key"], inputs["value"], requests)
        moe = replay.moe(None, None)
        result, _ = replay.run_post_attention(requests, attention, value, attn_in, residual, moe, None)
        routes.append(moe.records[-1].vllm_ids)
    final_hidden = result.outputs[0]
    logits = F.linear(final_hidden[: len(token_ids) - 1], weights["lm_head.weight"])
    return torch.stack(routes, dim=1), logits


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--capture", required=True, help="a trainer capture whose first sequence is the prompt")
    parser.add_argument("--full-output-code", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    capture = torch.load(stdlib_io.BytesIO(io.read_bytes(args.capture)), map_location="cpu", weights_only=False)
    mask = capture["attention_mask"][0].bool()
    token_ids = capture["sequences"][0][mask].tolist()
    with tempfile.TemporaryDirectory() as directory:
        weights = truncated_export(args.model, args.layers, Path(directory))
        served = serve(Path(directory) / "model", token_ids)
        config = json.loads((Path(directory) / "model" / "config.json").read_text())
    shape = GrugShape.from_config(config)
    with vllm_config_context():
        routes, logits = replay_model(served["code"], shape, weights, token_ids)
    served_routes = torch.as_tensor(served["routes"]).to(routes.device)
    replayed = routes[: served_routes.shape[0]].to(served_routes.dtype)
    compared = total = equal = 0
    for position, values in served["logits"].items():
        for token, value in values.items():
            total += 1
            replay_value = logits[position - 1, token].float().item()
            equal += replay_value == value
            compared += 1
    result = {
        "prompt_tokens": len(token_ids),
        "routes_shape": list(served_routes.shape),
        "routes_equal_fraction": (replayed == served_routes).float().mean().item(),
        "routes_rows_all_equal": bool(torch.equal(replayed, served_routes)),
        "logits_compared": compared,
        "logits_equal_fraction": equal / max(total, 1),
        "piece_diff_against_full_model": diff_pieces(served["code"], args.full_output_code),
        "best_configs": sum(1 for relative in served["code"] if relative.endswith(".best_config")),
        "engine_settings": ENGINE_SETTINGS,
    }
    payload = json.dumps(result, indent=1, sort_keys=True, default=str)
    io.write_bytes_atomic(join_resource_path(args.output, "validation.json"), payload.encode())
    for relative, text in served["code"].items():
        io.write_bytes_atomic(join_resource_path(args.output, "output-code", relative), text.encode())
    print(payload, flush=True)
    passed = result["routes_rows_all_equal"] and result["logits_equal_fraction"] == 1.0
    print(("PASS" if passed else "FAIL") + " replay validation", args.output, flush=True)


if __name__ == "__main__":
    main()
