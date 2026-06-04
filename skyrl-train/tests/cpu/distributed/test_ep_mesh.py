"""Stage 4a — expert-parallel device-mesh construction (CPU gate G4-1 / G4-0-mesh).

Gates (scope §6):
  G4-1        create_device_mesh(4, 2, ep_size=2) → shape (1, 2, 2) with dim names
              ["ddp", "ep", "fsdp"]; divisibility asserts fire on bad combos.
  G4-0-mesh   ep_size==1 (default) → UNCHANGED 1-D ["fsdp"] / 2-D ["ddp","fsdp"]
              mesh, byte-identical to the pre-EP path.

The 3-D happy path needs a real (gloo) device mesh whose world size equals the
mesh-shape product, so the valid-shape assertions run inside a 4-rank gloo spawn
on CPU. The divisibility asserts fire BEFORE init_device_mesh, so the negative
cases need no process group.

Run::

    uv run --isolated --extra dev --extra ep pytest tests/cpu/distributed/test_ep_mesh.py
    # or directly (no pytest): python tests/cpu/distributed/test_ep_mesh.py
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import pytest
except ImportError:  # pytest absent on cluster envs — direct invocation still works
    pytest = None

from skyrl_train.distributed.fsdp_utils import create_device_mesh, get_sharding_strategy


# --------------------------------------------------------------------------- #
# G4-1 — divisibility asserts (no process group needed; fire pre-mesh)         #
# --------------------------------------------------------------------------- #


def test_ep_mesh_divisibility_asserts():
    # world_size not divisible by ep_size.
    raised = False
    try:
        create_device_mesh(world_size=6, fsdp_size=2, ep_size=4)
    except AssertionError:
        raised = True
    assert raised, "ep_size not dividing world_size must raise"

    # ep_size * fsdp_size > world_size (ddp would be 0 / non-integer).
    raised = False
    try:
        create_device_mesh(world_size=4, fsdp_size=4, ep_size=2)
    except AssertionError:
        raised = True
    assert raised, "ep_size*fsdp_size not dividing world_size must raise"
    print("[G4-1] divisibility asserts fire on bad combos: PASS")


# --------------------------------------------------------------------------- #
# G4-1 — valid 3-D shape + names (inside a 4-rank gloo spawn)                  #
# --------------------------------------------------------------------------- #


def _mesh_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29555"
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # G4-1: ep_size=2, fsdp_size=2 on 4 ranks → (ddp=1, ep=2, fsdp=2).
        mesh = create_device_mesh(world_size=4, fsdp_size=2, ep_size=2, device_type="cpu")
        assert tuple(mesh.mesh.shape) == (1, 2, 2), f"shape={tuple(mesh.mesh.shape)} != (1,2,2)"
        assert mesh.mesh_dim_names == ("ddp", "ep", "fsdp"), f"names={mesh.mesh_dim_names}"
        # submeshes addressable
        assert mesh["ep"].size() == 2
        assert mesh["fsdp"].size() == 2
        # 3-D mesh maps to HYBRID_SHARD (relaxed == 2 → in (2,3)).
        from torch.distributed.fsdp import ShardingStrategy

        assert get_sharding_strategy(mesh) == ShardingStrategy.HYBRID_SHARD

        # G4-0-mesh: ep_size=1 is byte-identical to today's 2-D mesh.
        mesh_2d = create_device_mesh(world_size=4, fsdp_size=2, ep_size=1, device_type="cpu")
        assert mesh_2d.mesh_dim_names == ("ddp", "fsdp"), f"ep=1 names={mesh_2d.mesh_dim_names}"
        assert tuple(mesh_2d.mesh.shape) == (2, 2)

        # G4-0-mesh: ep_size=1, fsdp_size=-1 → 1-D ["fsdp"].
        mesh_1d = create_device_mesh(world_size=4, fsdp_size=-1, ep_size=1, device_type="cpu")
        assert mesh_1d.mesh_dim_names == ("fsdp",), f"1-D names={mesh_1d.mesh_dim_names}"
        if rank == 0:
            print("[G4-1] 3-D mesh (1,2,2) names ['ddp','ep','fsdp'] + submeshes: PASS")
            print("[G4-0-mesh] ep_size=1 byte-identical 2-D/1-D mesh: PASS")
    finally:
        dist.destroy_process_group()


def test_ep_mesh_shape_and_names():
    world_size = 4
    mp.spawn(_mesh_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    test_ep_mesh_divisibility_asserts()
    test_ep_mesh_shape_and_names()
    print("ALL PASS")
