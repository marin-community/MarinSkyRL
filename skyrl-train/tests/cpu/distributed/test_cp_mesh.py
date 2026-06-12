"""Stage 3 — context-parallel device-mesh construction (CPU gate).

Mirrors ``test_ep_mesh.py``. Validates that ``create_device_mesh`` gains a ``cp``
dim with the load-bearing dim-order contract ``ddp < fsdp < cp < ep``, while the
flag-off path (``cp_size==1``) emits the byte-identical mesh of today (G1).

Invariants (Stage-3 scope §"Invariants tests must assert"):
  1. create_device_mesh(world=4, fsdp=4, cp=1) ⇒ dims/shape UNCHANGED vs today
     (no ``cp`` dim emitted) — G1.
  2. cp=2 ⇒ dims include ``cp`` in order ["ddp","fsdp","cp"], numel==world.
  3. cp=2, ep=2 ⇒ ["ddp","fsdp","cp","ep"], fsdp idx < cp idx < ep idx, numel==world.
  4. Bad factorization (cp ∤ world / numel mismatch) → AssertionError.
  5. cp_group world size == cp_size; ranks within a cp group are contiguous and
     disjoint across cp groups (ring-neighbor correctness).

The shape/group happy paths need a real (gloo) device mesh whose world size equals
the mesh-shape product, so they run inside an N-rank gloo spawn on CPU. The
divisibility asserts fire BEFORE init_device_mesh, so the negative cases need no
process group.

Run::

    apptainer exec <sif> python -m pytest tests/cpu/distributed/test_cp_mesh.py -v \
        -p no:cacheprovider --confcutdir tests/cpu/distributed
    # or directly (no pytest): python tests/cpu/distributed/test_cp_mesh.py
"""

import os

import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import pytest
except ImportError:  # pytest absent on cluster envs — direct invocation still works
    pytest = None

import skyrl_train.distributed.fsdp_utils as fsdp_utils
from skyrl_train.distributed.fsdp_utils import create_device_mesh


# --------------------------------------------------------------------------- #
# Provenance — confirm we import the worktree copy, not the baked /opt/SkyRL.  #
# (The SIF bakes SkyRL at /opt/SkyRL and shadows bare `python`; PYTHONPATH must #
#  point at the worktree skyrl-train so the edited code is the one under test.) #
# --------------------------------------------------------------------------- #


def test_import_provenance_not_opt_skyrl():
    src = os.path.realpath(fsdp_utils.__file__)
    print(f"[provenance] create_device_mesh imported from: {src}")
    assert not src.startswith("/opt/SkyRL"), (
        f"fsdp_utils imported from baked /opt/SkyRL ({src}); set "
        f"PYTHONPATH=<worktree>/skyrl-train so the edited code is under test"
    )
    # The function must carry the Stage-3 cp_size parameter.
    import inspect

    params = inspect.signature(create_device_mesh).parameters
    assert "cp_size" in params, f"create_device_mesh missing cp_size param: {list(params)}"


# --------------------------------------------------------------------------- #
# #4 — divisibility asserts (no process group needed; fire pre-mesh)          #
# --------------------------------------------------------------------------- #


def test_cp_mesh_divisibility_asserts():
    # cp_size does not divide world_size.
    raised = False
    try:
        create_device_mesh(world_size=6, fsdp_size=1, cp_size=4)
    except AssertionError:
        raised = True
    assert raised, "cp_size not dividing world_size must raise"

    # fsdp_size * cp_size > world_size (ddp would be 0 / non-integer).
    raised = False
    try:
        create_device_mesh(world_size=4, fsdp_size=4, cp_size=2)
    except AssertionError:
        raised = True
    assert raised, "fsdp_size*cp_size not dividing world_size must raise"

    # cp*ep*fsdp numel mismatch.
    raised = False
    try:
        create_device_mesh(world_size=4, fsdp_size=2, cp_size=2, ep_size=2)
    except AssertionError:
        raised = True
    assert raised, "fsdp*cp*ep > world_size must raise"
    print("[#4] divisibility / numel-mismatch asserts fire on bad combos: PASS")


# --------------------------------------------------------------------------- #
# #1,#2,#3 — shape + dim-order contract (inside a gloo spawn)                  #
# --------------------------------------------------------------------------- #


def _shape_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29557"
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # #1 (G1): cp_size=1 on a single fsdp group → UNCHANGED 1-D ["fsdp"].
        m_1d = create_device_mesh(world_size=4, fsdp_size=4, cp_size=1, device_type="cpu")
        assert m_1d.mesh_dim_names == ("fsdp",), f"#1 1-D names={m_1d.mesh_dim_names}"
        assert tuple(m_1d.mesh.shape) == (4,), f"#1 1-D shape={tuple(m_1d.mesh.shape)}"
        assert "cp" not in m_1d.mesh_dim_names, "#1 cp dim must NOT be emitted at cp_size=1"

        # #1 (G1): cp_size=1 with fsdp_size=2 → UNCHANGED 2-D ["ddp","fsdp"].
        m_2d = create_device_mesh(world_size=4, fsdp_size=2, cp_size=1, device_type="cpu")
        assert m_2d.mesh_dim_names == ("ddp", "fsdp"), f"#1 2-D names={m_2d.mesh_dim_names}"
        assert tuple(m_2d.mesh.shape) == (2, 2), f"#1 2-D shape={tuple(m_2d.mesh.shape)}"
        assert "cp" not in m_2d.mesh_dim_names

        # #2: cp_size=2 → ["ddp","fsdp","cp"], numel==world.
        m_cp = create_device_mesh(world_size=4, fsdp_size=2, cp_size=2, device_type="cpu")
        assert m_cp.mesh_dim_names == ("ddp", "fsdp", "cp"), f"#2 names={m_cp.mesh_dim_names}"
        assert tuple(m_cp.mesh.shape) == (1, 2, 2), f"#2 shape={tuple(m_cp.mesh.shape)}"
        assert m_cp.mesh.numel() == world_size, f"#2 numel={m_cp.mesh.numel()} != {world_size}"
        assert m_cp["cp"].size() == 2
        names = m_cp.mesh_dim_names
        assert names.index("fsdp") < names.index("cp"), "#2 fsdp must precede cp"

        if rank == 0:
            print("[#1] cp_size=1 → byte-identical 1-D/2-D mesh (no cp dim): PASS")
            print("[#2] cp_size=2 → ['ddp','fsdp','cp'] numel==world: PASS")
    finally:
        dist.destroy_process_group()


def _shape_worker_cp_ep(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29558"
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # #3: cp=2, ep=2, fsdp=2 on 8 ranks → (ddp=1, fsdp=2, cp=2, ep=2).
        mesh = create_device_mesh(world_size=8, fsdp_size=2, cp_size=2, ep_size=2, device_type="cpu")
        assert mesh.mesh_dim_names == ("ddp", "fsdp", "cp", "ep"), f"#3 names={mesh.mesh_dim_names}"
        assert tuple(mesh.mesh.shape) == (1, 2, 2, 2), f"#3 shape={tuple(mesh.mesh.shape)}"
        assert mesh.mesh.numel() == world_size, f"#3 numel={mesh.mesh.numel()} != {world_size}"
        # Ordering contract: fsdp < cp < ep (load-bearing for fsdp-before-ep expert composition).
        names = mesh.mesh_dim_names
        assert (
            names.index("fsdp") < names.index("cp") < names.index("ep")
        ), f"#3 dim order must be fsdp < cp < ep; got {names}"
        assert mesh["fsdp"].size() == 2 and mesh["cp"].size() == 2 and mesh["ep"].size() == 2
        if rank == 0:
            print("[#3] cp=2,ep=2 → ['ddp','fsdp','cp','ep'] fsdp<cp<ep numel==world: PASS")
    finally:
        dist.destroy_process_group()


def _cp_group_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29559"
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # #5: cp_size=2 on 4 ranks → (ddp=1, fsdp=2, cp=2). The cp group this rank
        # belongs to must have world size == cp_size and contiguous, disjoint members.
        cp_size = 2
        mesh = create_device_mesh(world_size=4, fsdp_size=2, cp_size=cp_size, device_type="cpu")
        cp_group = mesh["cp"].get_group()
        assert (
            dist.get_world_size(cp_group) == cp_size
        ), f"#5 cp_group world_size={dist.get_world_size(cp_group)} != cp_size={cp_size}"
        # Gather every rank's cp-group global-rank membership and verify the partition.
        members = dist.get_process_group_ranks(cp_group)
        assert len(members) == cp_size, f"#5 cp_group members={members} != {cp_size}"
        assert rank in members, f"#5 own rank {rank} not in its cp_group {members}"
        # Contiguity: with mesh layout (ddp=1, fsdp=2, cp=2) the cp dim is the last
        # (fastest-varying) axis, so each cp group is a contiguous run of `cp_size`
        # consecutive global ranks. members are sorted by torch.
        assert members == list(
            range(members[0], members[0] + cp_size)
        ), f"#5 cp_group {members} is not a contiguous run of {cp_size} ranks"
        assert members[0] % cp_size == 0, f"#5 cp_group base {members[0]} not aligned to {cp_size}"

        # Disjointness across groups: collect all groups' member tuples and confirm a partition.
        all_members = [None] * world_size
        dist.all_gather_object(all_members, tuple(members))
        unique = set(all_members)
        union = set()
        for grp in unique:
            assert union.isdisjoint(grp), f"#5 cp groups overlap: {grp} vs {union}"
            union |= set(grp)
        assert union == set(range(world_size)), f"#5 cp groups do not partition all ranks: {union}"
        if rank == 0:
            print(f"[#5] cp_group size=={cp_size}, contiguous + disjoint partition " f"{sorted(unique)}: PASS")
    finally:
        dist.destroy_process_group()


def test_cp_mesh_shape_and_names():
    mp.spawn(_shape_worker, args=(4,), nprocs=4, join=True)


def test_cp_ep_mesh_ordering():
    mp.spawn(_shape_worker_cp_ep, args=(8,), nprocs=8, join=True)


def test_cp_group_contiguous_disjoint():
    mp.spawn(_cp_group_worker, args=(4,), nprocs=4, join=True)


if __name__ == "__main__":
    test_import_provenance_not_opt_skyrl()
    test_cp_mesh_divisibility_asserts()
    test_cp_mesh_shape_and_names()
    test_cp_ep_mesh_ordering()
    test_cp_group_contiguous_disjoint()
    print("ALL PASS")
