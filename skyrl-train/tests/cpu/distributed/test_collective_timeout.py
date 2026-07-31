import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl_train.distributed.fsdp_utils import create_device_mesh


def _mismatched_collective_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["SKYRL_WORKER_NCCL_TIMEOUT_IN_S"] = "1"
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    try:
        mesh = create_device_mesh(world_size, fsdp_size=2, ep_size=2, device_type="cpu")
        group = mesh["ep"].get_group() if rank in (0, 3) else mesh["fsdp"].get_group()

        started_at = time.monotonic()
        try:
            dist.all_reduce(torch.ones(1), group=group)
        except RuntimeError as error:
            assert "timed out" in str(error).lower()
            assert time.monotonic() - started_at < 5, "collective fell through to the 10-second WORLD timeout"
        else:
            raise AssertionError("rank-divergent collective did not honor the device-mesh timeout")
    finally:
        dist.destroy_process_group()


def test_rank_divergence_times_out_on_device_mesh_groups(unused_tcp_port):
    mp.spawn(_mismatched_collective_worker, args=(4, unused_tcp_port), nprocs=4, join=True)
