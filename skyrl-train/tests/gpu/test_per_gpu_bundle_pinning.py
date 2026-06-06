"""
GPU smoke test for the GH200 per-GPU-bundle device-pinning fix.

Reproduces the "4 policy actors per node" placement WITHOUT a full 80B run
(uses Qwen3-0.6B), and proves each policy actor lands on a DISTINCT physical
GPU after the per-GPU {GPU:1}-bundle policy placement group + ray.get_gpu_ids()
pinning. The pass criterion is: across all ranks, the set of
torch.cuda.current_device() indices and the set of physical GPU UUIDs are each
fully distinct (no two ranks on the same physical GPU), and a node's GPUs stay
contiguous in rank order (NVLink locality).

This is the cheap multi-node analogue of the 80B init-OOM repro: the bug was 4
ranks materializing onto GPU 0; here we assert non-collision deterministically.

Smallest useful shape is 2 nodes x 4 GPU (8 GPUs). Run several times to show
the distinctness holds regardless of Ray bundle tiling:

  uv run --isolated --extra dev -- pytest tests/gpu/test_per_gpu_bundle_pinning.py -q

Requires a >= 2-node, 4-GPU/node allocation (the bug only manifests multi-node).
"""

import os

import hydra
import pytest
import ray
import torch
from omegaconf import DictConfig

from skyrl_train.utils.utils import (
    get_ray_pg_ready_with_timeout,
    policy_spread_bundles,
    policy_per_gpu_bundles_enabled,
)
from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.entrypoints.main_base import config_dir


MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_NODES = int(os.environ.get("SMOKE_NUM_NODES", "2"))
GPUS_PER_NODE = int(os.environ.get("SMOKE_GPUS_PER_NODE", "4"))


def _make_cfg() -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.generator.weight_sync_backend = "nccl"
    cfg.trainer.strategy = "fsdp2"
    # Placement-only smoke: use the console tracking backend so validate_cfg()
    # does not require WANDB_API_KEY (the base config defaults logger="wandb").
    cfg.trainer.logger = "console"
    # Disaggregated, no-ref, per-GPU-bundle dedicated policy PG (the fix path).
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.placement.policy_strict_spread_pg = True
    cfg.trainer.placement.policy_per_gpu_bundles = True
    cfg.trainer.placement.policy_num_nodes = NUM_NODES
    cfg.trainer.placement.policy_num_gpus_per_node = GPUS_PER_NODE
    validate_cfg(cfg)
    return cfg


def _probe_device_id(self):
    """Run inside each actor: apply the fix's pinning decision and return the
    resulting physical placement + NUMA binding.

    DistributedTorchRayActor.__init__ (the fix) only *writes* os.environ[
    "LOCAL_RANK"] (= resolve_pinned_local_rank()); the actual
    torch.cuda.set_device() + _set_numa_affinity() run later in the strategy's
    setup_distributed(). This probe reproduces exactly that production sequence
    (set_device(LOCAL_RANK) then _set_numa_affinity(rank)) so what we measure is
    precisely where the fix's LOCAL_RANK lands the device + CPU/NUMA mask.
    """
    import torch as _torch
    from skyrl_train.utils.utils import get_physical_gpu_id as _uuid

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", "-1"))

    # Pin exactly as setup_distributed() does (consumes the fix's LOCAL_RANK).
    _torch.cuda.set_device(local_rank)
    # Apply the same NUMA binding the worker does, keyed off the physical id.
    try:
        self._set_numa_affinity(rank)
    except Exception:
        pass

    dev = _torch.cuda.current_device()
    # CPU affinity mask the NUMA bind produced (evidence the bind ran + is
    # distinct per physical GPU socket).
    try:
        cpu_affinity = sorted(os.sched_getaffinity(0))
        cpu_affinity_summary = (
            f"{cpu_affinity[0]}-{cpu_affinity[-1]} (n={len(cpu_affinity)})"
            if cpu_affinity else "none"
        )
    except Exception:
        cpu_affinity_summary = "unavailable"
    # GPU's hardware NUMA node (sysfs), to show device + NUMA agree.
    gpu_numa_node = None
    try:
        from skyrl_train.utils.numa import _enumerate_gpus_from_sysfs

        gpu_map = _enumerate_gpus_from_sysfs()
        if gpu_map is not None:
            gpu_numa_node = gpu_map.get(local_rank)
    except Exception:
        pass

    return {
        "rank": rank,
        "local_rank": os.environ.get("LOCAL_RANK"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "current_device": dev,
        "physical_uuid": _uuid(),
        "gpu_numa_node": gpu_numa_node,
        "cpu_affinity": cpu_affinity_summary,
        "node_ip": ray.util.get_node_ip_address(),
    }


@pytest.mark.parametrize("trial", range(int(os.environ.get("SMOKE_TRIALS", "3"))))
def test_per_gpu_bundle_distinct_physical_gpus(trial):
    """Each of NUM_NODES * GPUS_PER_NODE policy actors must pin a DISTINCT
    physical GPU. Repeated across trials to prove determinism vs Ray tiling."""
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    cfg = _make_cfg()
    assert policy_per_gpu_bundles_enabled(cfg) is True
    world_size = NUM_NODES * GPUS_PER_NODE

    bundles = policy_spread_bundles(cfg)
    assert len(bundles) == world_size, "per-GPU bundles must be one per GPU (== world_size)"
    pg = ray.util.placement_group(bundles, strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=120)

    policy = PPORayActorGroup(
        cfg,
        num_nodes=NUM_NODES,
        num_gpus_per_node=GPUS_PER_NODE,
        ray_actor_type=PolicyWorker,
        pg=pg,
        num_gpus_per_actor=1,
        colocate_all=False,
        pin_to_ray_gpu_id=True,  # the fix flag
    )

    # Bind the probe as a remote method on each actor handle.
    probes = ray.get([
        actor.__ray_call__.remote(_probe_device_id) for actor in policy._actor_handlers
    ])

    # --- Per-rank evidence (UUID / device / NUMA), printed for the report. ---
    print(f"\n=== trial {trial}: per-rank pinning evidence ===")
    for p in sorted(probes, key=lambda x: x["rank"]):
        print(
            f"  rank={p['rank']:>2} local_rank={p['local_rank']} "
            f"dev={p['current_device']} uuid={p['physical_uuid']} "
            f"gpu_numa={p['gpu_numa_node']} cpu_aff={p['cpu_affinity']} "
            f"node={p['node_ip']} CVD={p['cuda_visible_devices']}"
        )

    # --- Distinctness: no two ranks share a physical GPU on the same node. ---
    by_node = {}
    for p in probes:
        by_node.setdefault(p["node_ip"], []).append(p)

    for node_ip, node_probes in by_node.items():
        uuids = [p["physical_uuid"] for p in node_probes]
        devs = [p["current_device"] for p in node_probes]
        numas = [p["gpu_numa_node"] for p in node_probes]
        assert len(set(uuids)) == len(uuids), (
            f"node {node_ip}: ranks collided on the same physical GPU UUID: {node_probes}"
        )
        assert len(set(devs)) == len(devs), (
            f"node {node_ip}: ranks collided on the same torch device index: {node_probes}"
        )
        assert len(node_probes) == GPUS_PER_NODE, (
            f"node {node_ip}: expected {GPUS_PER_NODE} ranks, got {len(node_probes)}"
        )
        # NUMA: when sysfs NUMA nodes are resolvable, each rank's GPU must sit on
        # its own NUMA node (the fix keys NUMA off the physical id it pinned).
        if all(n is not None for n in numas):
            assert len(set(numas)) == len(numas), (
                f"node {node_ip}: ranks share a GPU NUMA node (wrong-socket bind): {node_probes}"
            )

    # --- NVLink locality: a node's ranks are contiguous in rank order. ---
    ranks_sorted = sorted(probes, key=lambda p: p["rank"])
    for n in range(NUM_NODES):
        group = ranks_sorted[n * GPUS_PER_NODE:(n + 1) * GPUS_PER_NODE]
        assert len(set(p["node_ip"] for p in group)) == 1, (
            f"rank group {n} spans multiple nodes (NVLink locality broken): {group}"
        )

    # Global distinctness (every rank on its own physical GPU cluster-wide).
    all_uuids = [p["physical_uuid"] for p in probes]
    assert len(set(all_uuids)) == world_size, f"global GPU collision: {probes}"

    for p in policy._actor_handlers:
        ray.kill(p)
    ray.util.remove_placement_group(pg)
