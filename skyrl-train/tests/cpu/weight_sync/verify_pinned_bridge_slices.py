"""Run the exact locked wheel's split/numbering functions against frozen views.

This CPU proof executes selected pure functions, not native Megatron collectives
or TE module construction. Live geometry and storage remain separate gates.
"""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import tomllib
from types import SimpleNamespace
import urllib.request
import zipfile

import torch

from skyrl_train.weight_sync.frozen_source_views import local_source_slices, source_view


def compile_function(node, namespace, filename):
    module = ast.fix_missing_locations(
        ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
            type_ignores=[],
        )
    )
    exec(compile(module, filename, "exec"), namespace)
    return namespace[node.name]


def verify(receipt_path):
    root = Path(__file__).parents[4]
    lock = tomllib.loads((root / "uv.lock").read_text())
    package = next(item for item in lock["package"] if item["name"] == "megatron-bridge")
    assert package["version"] == "0.5.0"
    wheel = package["wheels"][0]
    with urllib.request.urlopen(wheel["url"], timeout=30) as response:
        raw = response.read(wheel["size"] + 1)
    assert len(raw) == wheel["size"] and "sha256:" + hashlib.sha256(raw).hexdigest() == wheel["hash"]
    archive = zipfile.ZipFile(io.BytesIO(raw))
    prefix = "megatron/bridge/models/conversion/"
    texts = {name: archive.read(prefix + name).decode() for name in ("param_mapping.py", "model_bridge.py")}
    mapping_tree = ast.parse(texts["param_mapping.py"])
    qkv_node = next(
        node for node in mapping_tree.body if isinstance(node, ast.FunctionDef) and node.name == "split_qkv_weights"
    )
    split = compile_function(qkv_node, {"torch": torch}, prefix + "param_mapping.py")
    settings = SimpleNamespace(
        tensor_model_parallel_size=1,
        num_moe_experts=4,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        hidden_size=3,
    )
    qkv = torch.arange(48, dtype=torch.bfloat16).reshape(16, 3)
    qkv_mapping = type("QKVMapping", (), {})()
    qkv_mapping.hf_param = {"q": "q", "k": "k", "v": "v"}
    task = SimpleNamespace(param_weight=qkv, global_param_name="qkv.weight", mapping=qkv_mapping)
    slices, sources = local_source_slices([task], settings)
    for name, reference in zip(("q", "k", "v"), split(settings, qkv), strict=True):
        actual = torch.cat(
            [
                source_view(item, sources)
                for item in sorted(slices, key=lambda item: item.hf_offset)
                if item.hf_name == name
            ]
        )
        assert torch.equal(actual.view(torch.uint8), reference.reshape(-1).view(torch.uint8))
    gated_class = next(
        node for node in mapping_tree.body if isinstance(node, ast.ClassDef) and node.name == "GatedMLPMapping"
    )
    gated_node = next(
        node for node in gated_class.body if isinstance(node, ast.FunctionDef) and node.name == "megatron_to_hf"
    )
    native_gated = compile_function(gated_node, {"torch": torch}, prefix + "param_mapping.py")
    gated = torch.arange(24, dtype=torch.bfloat16).reshape(8, 3)
    mapping = type("GatedMLPMapping", (), {})()
    mapping.hf_param = {"gate": "gate", "up": "up"}
    mapping.tp_size = 1
    mapping.is_expert = False
    mapping.broadcast_from_pp_rank = lambda tensor, **kwargs: tensor
    mapping.maybe_dequantize = lambda tensor: tensor
    slices, sources = local_source_slices(
        [SimpleNamespace(param_weight=gated, global_param_name="mlp.weight", mapping=mapping)], settings
    )
    expected = native_gated(mapping, gated, None)
    assert all(
        torch.equal(source_view(item, sources).view(torch.uint8), expected[item.hf_name].reshape(-1).view(torch.uint8))
        for item in slices
    )
    bridge_tree = ast.parse(texts["model_bridge.py"])
    global_node = next(
        node
        for node in bridge_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_megatron_local_name_to_global"
    )
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    layer = SimpleNamespace(layer_number=10)
    namespace = {
        "re": re,
        "_get_pp_group": lambda models: group,
        "_get_ep_group": lambda models: group,
        "get_pg_size": lambda value: value.size(),
        "MegatronModule": SimpleNamespace,
        "get_module_and_param_from_name": lambda **kwargs: (None, layer),
    }
    global_name = compile_function(global_node, namespace, prefix + "model_bridge.py")
    actual_name = global_name([], settings, "decoder.layers.0.mlp.experts.linear_fc1.weight1", 0)
    assert actual_name == "decoder.layers.9.mlp.experts.linear_fc1.weight3"
    assert not torch.cuda.is_initialized()
    receipt = {
        "wheel_sha256": wheel["hash"].removeprefix("sha256:"),
        "wheel_bytes": len(raw),
        "source_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()},
        "qkv_every_byte": True,
        "gated_every_byte": True,
        "global_layer_and_expert_numbering": actual_name,
        "cuda_initialized": False,
        "scope": "exact locked wheel pure split/numbering functions; synthetic task mapping objects; no native layout or collective claim",
    }
    Path(receipt_path).write_text(json.dumps(receipt, indent=2) + "\n")
    print("PINNED_BRIDGE_FROZEN_SOURCE_VIEWS_PASS qkv_bytes=true gated_bytes=true global_numbering=true cuda=false")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    verify(parser.parse_args().receipt)
