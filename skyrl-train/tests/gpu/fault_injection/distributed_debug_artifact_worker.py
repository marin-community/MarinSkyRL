"""Two-rank worker used by the opt-in distributed-debug artifact contract."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

from skyrl_train.env_vars import write_process_manifest


def _write_receipt(artifact_root: Path, mode: str, rank: int, outcome: str) -> None:
    path = artifact_root / "runs" / f"{mode}.rank{rank}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"mode": mode, "rank": rank, "outcome": outcome}, sort_keys=True) + "\n")
    temporary.replace(path)


def main(mode: str) -> None:
    rank = int(os.environ["RANK"])
    artifact_root = Path(os.environ["SKYRL_DEBUG_ARTIFACT_DIR"])
    write_process_manifest(f"{mode}-rank", metadata={"mode": mode, "rank": rank})

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=12))
    tensor = torch.tensor([float(rank + 1)], device="cuda")
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    assert tensor.item() == 3.0
    if mode == "healthy":
        _write_receipt(artifact_root, mode, rank, "completed")
        dist.destroy_process_group()
        return

    tensor.fill_(float(rank + 1))
    if rank == 1:
        _write_receipt(artifact_root, mode, rank, "withheld-before-collective")
        time.sleep(180)  # deliberate fault input; torchrun must terminate this rank
        raise RuntimeError("withheld rank survived torchrun teardown")

    try:
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
    except Exception:
        _write_receipt(artifact_root, mode, rank, "collective-failed")
        raise
    raise RuntimeError("non-arrival collective completed unexpectedly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("healthy", "nonarrival"), required=True)
    main(parser.parse_args().mode)
