"""EP-REPRODUCING device-assignment smoke.

Exercises the ACTUAL failing placement of the Qwen3-Next-80B run at small
scale: a small MoE (Qwen1.5-MoE-A2.7B) built with EP=2 x FSDP=2 on 4 GH200
GPUs, through the DEDICATED per-GPU {GPU:1}-bundle policy PG with
colocate_all=False — i.e. the exact path that stacks ranks on physical GPU 0
in job 646698 (not the colocate_all=True shared-PG path the prior EP tests
used, which never reproduced the bug).

It then materializes the MoE on every EP×FSDP policy rank (the
fsdp2_load_full_state_dict `.to(device=current_device())` step that OOM'd) and
asserts every rank landed on a DISTINCT physical GPU (by UUID), with the EP
device-mesh coordinate reported.

PASS := DISTINCT_PHYSICAL_GPUS == world_size (no two ranks share a physical GPU)
        AND every rank built the 3-D EP mesh.

Env:
  SMOKE_FORCE_CVD_MASK = "1" | "0"   (toggle the fix; default "1")
  SMOKE_NUM_GPUS       = "4"         (world size; EP=2 x FSDP=2)
"""
import os
import sys
import socket

import ray
from omegaconf import OmegaConf

from skyrl_train.utils.utils import (
    initialize_ray,
    get_ray_pg_ready_with_timeout,
    get_reordered_bundle_indices,  # noqa: F401 (sanity import)
)
from ray.util.placement_group import placement_group
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.entrypoints.main_base import config_dir  # for default config load

MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen1.5-MoE-A2.7B-Chat")
NUM_GPUS = int(os.environ.get("SMOKE_NUM_GPUS", "4"))
EP = 2
FSDP = 2
FORCE_CVD_MASK = os.environ.get("SMOKE_FORCE_CVD_MASK", "1") == "1"


def build_cfg():
    import hydra
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")
    OmegaConf.set_struct(cfg, False)  # allow setting (possibly-new) keys
    # Trainer / model
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.policy.fsdp_config.expert_model_parallel_size = EP
    cfg.trainer.policy.fsdp_config.fsdp_size = FSDP
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    cfg.trainer.policy.fsdp_config.moe_grouped_gemm = True
    cfg.trainer.policy.fsdp_config.ep_comm_backend = "torch"
    cfg.trainer.policy.fsdp_config.cpu_offload = True
    cfg.trainer.policy.sequence_parallel_size = 1
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.gradient_checkpointing_use_reentrant = False
    cfg.trainer.flash_attn = True
    # Placement: disaggregated dedicated per-GPU-bundle policy PG (the real path)
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.placement.policy_strict_spread_pg = True
    cfg.trainer.placement.policy_per_gpu_bundles = True
    cfg.trainer.placement.policy_force_cvd_mask = FORCE_CVD_MASK
    # No ref model, no KL (so the policy PG is dedicated, mirroring the 80B run)
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    return cfg


def main():
    cfg = build_cfg()
    print(f"[smoke] MODEL={MODEL} EP={EP} FSDP={FSDP} world={NUM_GPUS} "
          f"force_cvd_mask={FORCE_CVD_MASK}", flush=True)

    initialize_ray(cfg)

    # Dedicated per-GPU {GPU:1} bundles, exactly like get_policy_pg() for the
    # per-GPU-bundle path. world_size bundles -> reorder path engaged.
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(NUM_GPUS)]
    pg = placement_group(bundles, strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=120)

    from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker

    policy = PPORayActorGroup(
        cfg,
        num_nodes=cfg.trainer.placement.policy_num_nodes,
        num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
        ray_actor_type=PolicyWorker,
        pg=pg,
        num_gpus_per_actor=1,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        pin_to_ray_gpu_id=True,
        force_cvd_mask=FORCE_CVD_MASK,
    )

    # This is the step that OOM'd in 646698 (fsdp2_load_full_state_dict ->
    # .to(device=current_device())). If ranks stack on GPU0 it OOMs here.
    ray.get(policy.async_init_model(MODEL))

    diags = ray.get(policy.async_run_ray_method("pass_through", "get_device_placement_diag"))
    diags = sorted(diags, key=lambda d: d["rank"])
    print("\n==== PER-RANK EP×FSDP DEVICE PLACEMENT ====", flush=True)
    for d in diags:
        print(d, flush=True)

    # Assertions
    seen = {}
    collision = False
    mesh_ok = True
    for d in diags:
        key = (d["host"], d["phys_uuid"])
        if key in seen:
            collision = True
            print(f"COLLISION: rank {d['rank']} and rank {seen[key]} share {key}", flush=True)
        seen[key] = d["rank"]
        if d.get("mesh_shape") is None or len(d.get("mesh_shape", ())) != 3:
            mesh_ok = False

    print(f"\nDISTINCT_PHYSICAL_GPUS={len(seen)} / {NUM_GPUS}", flush=True)
    print(f"EP_MESH_3D_ALL_RANKS={'YES' if mesh_ok else 'NO'}", flush=True)
    ok = (not collision) and mesh_ok and len(seen) == NUM_GPUS
    print("SMOKE_RESULT=" + ("PASS_DISTINCT_EP" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
