"""Disposable one-GPU component sweep for exact BF16 routed expert patches."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import torch
from skyrl_train.weight_sync.expert_block.sparse_experiment_codec import (
    Patch,
    changed_mask,
    pack_bitmap,
)
from skyrl_train.weight_sync.expert_block.sparse_experiment_codec import (
    apply as apply_patch,
)

MODEL_REPO = "marin-community/grug-67b-a2b-sft-s2-thinking-step630"
MODEL_REVISION = "6808fe5c219471517bd51df35addefd38ebebf89"
DENSITIES = (0.001, 0.01, 0.04, 0.1, 0.231647, 1 / 3, 0.5, 0.8)
SIZES = {
    "expert_fc2": 2560 * 1280,
    "expert_fc1": 2 * 2560 * 1280,
    "embedding": 128256 * 2560,
}


def _synchronize() -> None:
    torch.cuda.synchronize()


def _timed(operation):
    _synchronize()
    start = time.perf_counter()
    result = operation()
    _synchronize()
    return result, time.perf_counter() - start


def _sample(current: torch.Tensor, previous: torch.Tensor, destination: torch.Tensor, encoding: str) -> dict:
    destination.copy_(previous)
    _synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    stage_start = time.perf_counter()
    if encoding == "dense":
        _, apply_seconds = _timed(lambda: destination.copy_(current))
        detection = construction = packing = 0.0
        logical_bytes = current.numel() * current.element_size()
        changed = None
    else:
        mask, detection = _timed(lambda: changed_mask(current, previous))
        if encoding == "indices":
            positions, construction = _timed(lambda: mask.nonzero(as_tuple=False).view(-1).to(torch.int32))
            values, packing = _timed(lambda: current.view(-1).index_select(0, positions.to(torch.int64)))
        elif encoding == "bitmap":
            positions, construction = _timed(lambda: pack_bitmap(mask))
            values, packing = _timed(lambda: current.view(-1).masked_select(mask))
        else:
            raise ValueError(encoding)
        patch = Patch(encoding, current.numel(), positions, values)
        changed = patch.changed
        _, apply_seconds = _timed(lambda: apply_patch(destination, patch))
        logical_bytes = patch.payload_bytes
    total_seconds = time.perf_counter() - stage_start
    if not torch.equal(destination.view(torch.int16), current.view(torch.int16)):
        raise RuntimeError(f"Raw BF16 bytes differ after {encoding} application")
    return {
        "encoding": encoding,
        "changed": changed,
        "values": current.numel(),
        "logical_bytes": logical_bytes,
        "detect_seconds": detection,
        "construct_seconds": construction,
        "pack_seconds": packing,
        "apply_seconds": apply_seconds,
        "total_seconds": total_seconds,
        "gpu_allocated_before": allocated_before,
        "gpu_peak_allocated": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This sweep requires one CUDA GPU")
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(MODEL_REPO, "config.json", revision=MODEL_REVISION)
    config = json.loads(Path(config_path).read_text())
    if config["hidden_size"] != 2560 or config["intermediate_size"] != 1280 or config["vocab_size"] != 128256:
        raise RuntimeError("The pinned Grug model no longer matches the benchmark shapes")
    result = {
        "schema_version": 1,
        "source_commit": os.environ["SPARSE_EXPERIMENT_SOURCE_COMMIT"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_config": {key: config[key] for key in ("hidden_size", "intermediate_size", "vocab_size")},
        "torch_version": importlib.metadata.version("torch"),
        "gpu": torch.cuda.get_device_name(),
        "samples": [],
        "complete": False,
    }
    output_dir = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sparse-expert-microbench.json"
    try:
        for family, size in SIZES.items():
            for requested_density in DENSITIES:
                torch.manual_seed(1719)
                previous = torch.randn(size, dtype=torch.bfloat16, device="cuda")
                current = previous.clone()
                selected = torch.rand(size, device="cuda") < requested_density
                raw = current.view(torch.int16)
                raw[selected] = raw[selected] ^ 1
                destination = previous.clone()
                actual_changed = int(selected.sum().item())
                del selected
                # Warm each path once, then rotate order so clock drift cannot favor one codec.
                for encoding in ("dense", "indices", "bitmap"):
                    _sample(current, previous, destination, encoding)
                for repeat in range(4):
                    order = ("dense", "indices", "bitmap") if repeat % 2 == 0 else ("bitmap", "indices", "dense")
                    for encoding in order:
                        sample = _sample(current, previous, destination, encoding)
                        sample.update(
                            family=family,
                            requested_density=requested_density,
                            actual_density=actual_changed / size,
                            repeat=repeat,
                        )
                        result["samples"].append(sample)
                print(f"MICROBENCH_DONE family={family} values={size} density={actual_changed / size:.6f}", flush=True)
                output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
                del previous, current, destination
                torch.cuda.empty_cache()
        result["complete"] = True
    finally:
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
