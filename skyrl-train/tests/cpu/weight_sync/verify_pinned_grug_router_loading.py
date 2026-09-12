"""Execute pinned router allocation/loading bodies on CPU, without an expert engine."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn


REVISION = "fa50698a9a303f7282aa0e969f35717703de4911"


def qualify(checkout: Path) -> dict:
    files = {}

    def source(path):
        raw = subprocess.check_output(["git", "-C", str(checkout), "show", f"{REVISION}:{path}"])
        files[path] = hashlib.sha256(raw).hexdigest()
        return ast.parse(raw)

    parameter = source("vllm/model_executor/parameter.py")
    linear = source("vllm/model_executor/layers/linear.py")
    grug = source("vllm/model_executor/models/grugmoe.py")
    platform = ModuleType("vllm.platforms")
    platform.current_platform = SimpleNamespace(use_sync_weight_loader=lambda: False)
    previous = sys.modules.get("vllm.platforms")
    sys.modules["vllm.platforms"] = platform

    def attrs(tensor, values):
        for key, value in values.items():
            setattr(tensor, key, value)

    class ExpertPlaceholder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.arguments = kwargs

    namespace = {
        "torch": torch,
        "nn": nn,
        "Parameter": nn.Parameter,
        "UninitializedParameter": nn.UninitializedParameter,
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "set_weight_attrs": attrs,
        "FusedMoE": ExpertPlaceholder,
        "GrugMoeRouter": lambda **kwargs: SimpleNamespace(**kwargs),
        "is_pp_missing_parameter": lambda name, model: False,
        "default_weight_loader": lambda param, loaded: param.data.copy_(loaded),
    }

    def execute(nodes):
        module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(module, "<pinned-vllm-cpu-boundary>", "exec"), namespace)

    def cls(tree, name, *, bases=None, methods=None):
        node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name)
        node.decorator_list = []
        if bases is not None:
            node.bases = [ast.parse(base, mode="eval").body for base in bases]
        if methods is not None:
            node.body = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name in methods]
        return node

    try:
        execute(
            [
                cls(parameter, name)
                for name in ("BasevLLMParameter", "_ColumnvLLMParameter", "RowvLLMParameter", "ModelWeightParameter")
            ]
        )
        execute(
            [
                cls(linear, "UnquantizedLinearMethod", bases=["object"], methods={"create_weights"}),
                cls(linear, "LinearBase", bases=["nn.Module"], methods={"__init__"}),
                cls(linear, "ReplicatedLinear", methods={"__init__", "weight_loader"}),
                cls(grug, "GrugMoeMLP", methods={"__init__"}),
            ]
        )
        mapping = next(
            x
            for x in grug.body
            if isinstance(x, ast.AnnAssign) and getattr(x.target, "id", "") == "_EXPERT_WEIGHT_MAPPING"
        )
        expert_loader = next(
            x for x in grug.body if isinstance(x, ast.FunctionDef) and x.name == "_try_load_grug_expert_weight"
        )
        loader_class = cls(grug, "GrugMoeForCausalLM", bases=["nn.Module"], methods={"load_weights"})
        execute([mapping, expert_loader, loader_class])
        cfg = SimpleNamespace(hidden_dim=2560, num_experts=256, num_experts_per_token=4, intermediate_dim=1280)
        layer = namespace["GrugMoeMLP"](cfg, params_dtype=torch.bfloat16, prefix="model.layers.0.mlp")
        assert type(layer.router.weight).__name__ == "ModelWeightParameter"
        assert layer.router.weight.dtype == layer.router.bias.dtype == torch.float32
        assert tuple(layer.router.weight.shape) == (256, 2560)
        assert layer.router.weight.is_contiguous()
        source_bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16).repeat(10)
        wire = source_bits.view(torch.bfloat16).reshape(256, 2560)
        name = "model.layers.0.mlp.router.weight"
        model = SimpleNamespace(tie_word_embeddings=False, named_parameters=lambda: [(name, layer.router.weight)])
        loaded = namespace["GrugMoeForCausalLM"].load_weights(model, [(name, wire)])
        assert loaded == {name}
        expected_bits = source_bits.to(torch.int32) << 16
        assert torch.equal(layer.router.weight.view(torch.int32).reshape(-1), expected_bits)
        from skyrl_train.weight_sync.router_replay import compare_widened_router

        scratch = torch.empty(512 * 1024, dtype=torch.bool)
        compared = compare_widened_router(wire, layer.router.weight, scratch)
        assert compared.mismatches == 0 and compared.compared_bytes == wire.numel() * 4
        return {
            "revision": REVISION,
            "source_sha256": files,
            "parameter": name,
            "wire_shape": list(wire.shape),
            "wire_dtype": str(wire.dtype),
            "wire_bytes": wire.numel() * wire.element_size(),
            "installed_shape": list(layer.router.weight.shape),
            "installed_dtype": str(layer.router.weight.dtype),
            "installed_bytes": layer.router.weight.numel() * layer.router.weight.element_size(),
            "sharding": "TP1 replicated router; EP shards experts only",
            "all_bf16_patterns": 65536,
            "original_loader_widening_bit_exact": True,
            "cuda": False,
            "bounded_replay_compared_bytes": compared.compared_bytes,
            "scope": "Exact pinned router/linear/parameter allocation and model weight-loader bodies; expert engine and platform/TP runtime are CPU boundary substitutes, not a native receiver readback",
        }
    finally:
        if previous is None:
            sys.modules.pop("vllm.platforms", None)
        else:
            sys.modules["vllm.platforms"] = previous


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-checkout", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.vllm_checkout)
    args.receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print("PINNED_GRUG_ROUTER_LOADING_PASS wire=bf16 installed=fp32 patterns=65536 bit_exact_widening=true cuda=false")
