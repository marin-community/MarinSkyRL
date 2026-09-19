"""Compare direct bound-view application with the pinned vLLM checkpoint loader.

This isolated apply comparison uses a contiguous named parameter. It excludes
a live vLLM worker, NCCL, and Grug expert-name routing.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import time
from pathlib import Path

import torch
from skyrl_train.weight_sync.expert_block.sparse_experiment_codec import apply, bits, encode
from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    load_checkpoint_weight_patches,
)


class _NamedModel:
    def __init__(self, parameter: torch.Tensor):
        self.parameter = parameter

    def get_parameter(self, name: str) -> torch.Tensor:
        if name != "model.embed_tokens.weight":
            raise KeyError(name)
        return self.parameter

    def load_weights(self, weights):
        names = []
        for name, value in weights:
            self.get_parameter(name).copy_(value)
            names.append(name)
        return names


def _measure(operation) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _case(name: str, count: int, density: float) -> list[dict]:
    baseline = torch.zeros(count, dtype=torch.bfloat16, device="cuda")
    current = baseline.clone()
    changed = max(1, round(count * density))
    positions = torch.arange(changed, dtype=torch.int64, device="cuda") * (count - 1) // max(changed - 1, 1)
    current.index_fill_(0, positions, 1)
    patch = encode(current, baseline, "indices")
    destination = baseline.clone()
    model = _NamedModel(destination)
    upstream_patch = CheckpointWeightPatch(
        name="model.embed_tokens.weight",
        shape=(count,),
        dtype=torch.bfloat16,
        indices=patch.positions,
        values=patch.values,
    )
    operations = {
        "bound_view": lambda: apply(destination, patch),
        "vllm_checkpoint_loader": lambda: load_checkpoint_weight_patches(model, [upstream_patch]),
    }
    for operation in operations.values():
        destination.copy_(baseline)
        operation()
        torch.cuda.synchronize()
    rows = []
    for repeat in range(8):
        order = tuple(operations) if repeat % 2 == 0 else tuple(reversed(tuple(operations)))
        for path in order:
            destination.copy_(baseline)
            seconds = _measure(operations[path])
            if not bool(torch.equal(bits(destination), bits(current))):
                raise RuntimeError(f"Raw bits differ after {path}")
            rows.append(
                {
                    "name": name,
                    "values": count,
                    "density": density,
                    "changed": patch.changed,
                    "path": path,
                    "repeat": repeat,
                    "apply_seconds": seconds,
                    "byte_equal": True,
                }
            )
    return rows


def _edge_semantics() -> dict:
    baseline = torch.tensor([0.0, 2.0], dtype=torch.bfloat16, device="cuda")
    result = {}
    for label, index, raw_bits in (("negative_zero", 0, -32768), ("nan_payload", 1, 32705)):
        current = baseline.clone()
        current.view(torch.int16)[index] = raw_bits
        patch = encode(current, baseline, "indices")
        direct = baseline.clone()
        apply(direct, patch)
        result[f"direct_{label}_byte_equal"] = bool(torch.equal(bits(direct), bits(current)))
        if not result[f"direct_{label}_byte_equal"]:
            raise RuntimeError(f"Direct apply changed {label} bits")
        destination = baseline.clone()
        model = _NamedModel(destination)
        upstream_patch = CheckpointWeightPatch(
            name="model.embed_tokens.weight",
            shape=(baseline.numel(),),
            dtype=torch.bfloat16,
            indices=patch.positions,
            values=patch.values,
        )
        try:
            load_checkpoint_weight_patches(model, [upstream_patch])
        except ValueError as error:
            result[f"vllm_{label}_error"] = str(error)
        else:
            result[f"vllm_{label}_byte_equal"] = bool(torch.equal(bits(destination), bits(current)))
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires one CUDA GPU")
    result = {
        "schema_version": 1,
        "source_commit": os.environ["SPARSE_EXPERIMENT_SOURCE_COMMIT"],
        "gpu": torch.cuda.get_device_name(),
        "torch_version": importlib.metadata.version("torch"),
        "vllm_version": importlib.metadata.version("vllm"),
        "samples": [],
        "edge_semantics": _edge_semantics(),
        "complete": False,
    }
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp")) / "sparse-expert-apply-compare.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, count in (("expert_fc2", 2560 * 1280), ("embedding", 128256 * 2560)):
            for density in (0.01, 0.231647):
                result["samples"].extend(_case(name, count, density))
                output.write_text(json.dumps(result, indent=2, sort_keys=True))
                print(f"SPARSE_APPLY_COMPARE_DONE {name} density={density}", flush=True)
        result["complete"] = True
    finally:
        output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
