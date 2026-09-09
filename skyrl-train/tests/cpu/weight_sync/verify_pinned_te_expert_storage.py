"""Execute the locked TE allocation loop on CPU; this does not initialize its CUDA runtime."""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tomllib
import zipfile

import torch


class NativeParameterBoundary(torch.nn.Module):
    def register_parameter(self, name, parameter, **metadata):
        super().register_parameter(name, parameter)


def verify(wheel: Path, receipt: Path) -> None:
    root = Path(__file__).parents[4]
    package = next(
        item
        for item in tomllib.loads((root / "uv.lock").read_text())["package"]
        if item["name"] == "transformer-engine"
    )
    assert package["version"] == "2.11.0"
    pinned = package["wheels"][0]
    raw = wheel.read_bytes()
    assert len(raw) == pinned["size"] and hashlib.sha256(raw).hexdigest() == pinned["hash"].removeprefix("sha256:")
    source = zipfile.ZipFile(io.BytesIO(raw)).read("transformer_engine/pytorch/module/grouped_linear.py")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GroupedLinear")
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assert "single_grouped_weight" not in {arg.arg for arg in constructor.args.args}
    loop = next(
        node
        for node in constructor.body
        if isinstance(node, ast.For) and "self.register_parameter" in ast.unparse(node)
    )
    module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    target = NativeParameterBoundary()
    target.num_gemms, target.out_features, target.in_features = 3, 4, 2
    target.use_bias = False
    target._offsets = {"weight": 0}
    target._num_fp8_tensors_per_gemm = {"fwd": 1}
    namespace = {
        "self": target,
        "torch": torch,
        "device": "cpu",
        "params_dtype": torch.bfloat16,
        "init_method": None,
        "get_rng_state_tracker": None,
    }
    exec(compile(module, "<locked-te211-allocation-loop>", "exec"), namespace)
    parameters = dict(target.named_parameters())
    assert list(parameters) == ["weight0", "weight1", "weight2"]
    assert all(
        tensor.shape == (4, 2) and tensor.stride() == (2, 1) and tensor.is_contiguous()
        for tensor in parameters.values()
    )
    assert len({tensor.data_ptr() for tensor in parameters.values()}) == 3
    evidence = {
        "wheel_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_url": pinned["url"],
        "parameters": {
            name: {"shape": list(tensor.shape), "stride": list(tensor.stride())} for name, tensor in parameters.items()
        },
        "single_grouped_weight_constructor_option": False,
        "scope": "Exact locked allocation loop and native CPU Parameter registration; full constructor/CUDA execution excluded",
    }
    receipt.write_text(json.dumps(evidence, indent=2) + "\n")
    print("LOCKED_TE211_PER_EXPERT_STORAGE_PASS matrices=3 contiguous=true distinct_storage=true cuda=false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    verify(args.wheel, args.receipt)


if __name__ == "__main__":
    main()
