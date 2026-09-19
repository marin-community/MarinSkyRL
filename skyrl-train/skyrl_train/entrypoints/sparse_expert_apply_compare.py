"""Compare direct bound-view patch application with vLLM's sparse apply method.

This isolated kernel comparison uses a contiguous named parameter, as vLLM's
sparse NCCL engine requires. It excludes lookup in a live vLLM worker, NCCL,
and Grug expert-name routing, which the current vLLM sparse engine lacks.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import time
from pathlib import Path

import torch
from skyrl_train.weight_sync.expert_block.sparse_experiment_codec import apply, bits, encode
from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseNCCLWeightTransferEngine, SparseWeightPatch


class _NamedModel:
    def __init__(self, parameter: torch.Tensor):
        self.parameter = parameter

    def get_parameter(self, name: str) -> torch.Tensor:
        if name != "model.embed_tokens.weight":
            raise KeyError(name)
        return self.parameter


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
    engine = type("ApplyOnly", (), {"model": _NamedModel(destination)})()
    upstream_patch = SparseWeightPatch(
        name="model.embed_tokens.weight", indices=patch.positions, values=patch.values, full_shape=(count,)
    )
    operations = {
        "bound_view": lambda: apply(destination, patch),
        "vllm_sparse_apply": lambda: SparseNCCLWeightTransferEngine._apply_patch(engine, upstream_patch),
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
