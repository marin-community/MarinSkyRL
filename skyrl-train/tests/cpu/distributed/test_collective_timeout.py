import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.cpu.util import gloo_process_group
from tests.distributed_runtime_constants import CPU_TEST_PROCESS_GROUP_TIMEOUT_SECONDS

from skyrl_train.distributed.fsdp_utils import create_device_mesh


MESH_PROCESS_GROUP_TIMEOUT_SECONDS = 1


def _mismatched_collective_worker(rank: int, world_size: int, port: int) -> None:
    with gloo_process_group(rank, world_size, port, timeout_seconds=CPU_TEST_PROCESS_GROUP_TIMEOUT_SECONDS):
        mesh = create_device_mesh(
            world_size,
            fsdp_size=2,
            ep_size=2,
            device_type="cpu",
            timeout_seconds=MESH_PROCESS_GROUP_TIMEOUT_SECONDS,
        )
        group = mesh["ep"].get_group() if rank in (0, 3) else mesh["fsdp"].get_group()

        started_at = time.monotonic()
        try:
            dist.all_reduce(torch.ones(1), group=group)
        except RuntimeError as error:
            elapsed_seconds = time.monotonic() - started_at
            assert "timed out" in str(error).lower(), f"collective failed without a timeout: {error}"
            assert elapsed_seconds >= MESH_PROCESS_GROUP_TIMEOUT_SECONDS / 2, (
                f"collective failed before its mesh deadline: {elapsed_seconds:.2f}s"
            )
            assert elapsed_seconds < CPU_TEST_PROCESS_GROUP_TIMEOUT_SECONDS / 2, (
                f"collective fell through to the {CPU_TEST_PROCESS_GROUP_TIMEOUT_SECONDS}-second WORLD timeout"
            )
        else:
            raise AssertionError("rank-divergent collective did not honor the device-mesh timeout")


def test_rank_divergence_times_out_on_device_mesh_groups(unused_tcp_port):
    mp.spawn(_mismatched_collective_worker, args=(4, unused_tcp_port), nprocs=4, join=True)
