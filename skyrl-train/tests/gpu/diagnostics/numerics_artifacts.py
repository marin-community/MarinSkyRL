from __future__ import annotations

import atexit
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import pytest


NONZERO_FLOOR = 1e-30
ARTIFACT_ROOT = Path(os.environ.get("SKYRL_NUMERICS_ARTIFACT_DIR", "backend-numerics-artifacts"))


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device


def artifact_directory(test_name: str) -> Path:
    path = ARTIFACT_ROOT / test_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_rows(test_name: str, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    directory = artifact_directory(test_name)
    fieldnames = sorted({key for row in rows for key in row})
    with (directory / "discrepancies.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "metadata": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            **metadata,
        },
        "rows": rows,
    }
    (directory / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def require_cuda_gpus(count: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    visible = torch.cuda.device_count()
    if visible < count:
        pytest.skip(f"requires {count} visible GPUs, found {visible}")


def _destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def initialize_distributed() -> DistributedContext:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
        atexit.register(_destroy_process_group)
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return DistributedContext(rank, dist.get_world_size(), torch.device("cuda", local_rank))


def require_distributed_world_size(world_size: int) -> DistributedContext:
    require_cuda_gpus(world_size)
    context = initialize_distributed()
    if context.world_size != world_size:
        pytest.skip(f"requires torchrun world size {world_size}, found {context.world_size}")
    return context
