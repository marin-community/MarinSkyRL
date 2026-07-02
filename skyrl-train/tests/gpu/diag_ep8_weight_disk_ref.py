"""EP=8 CROSS-NODE, NON-CIRCULAR weight-equality assert for the FSDP->vLLM MoE gather.

THE DECISIVE MEASUREMENT. Brings up ONLY the FSDP grouped+EP policy worker at the
PROD MoE factorization (EP=8 x FSDP=2 = 16 GPU) laid out 4 nodes x 4 GPU so the 8
EP ranks of a group STRADDLE >=2 physical nodes (verified, not assumed). NO inference
engine, NO rollout, NO Daytona, NO engine-readback. At gs0 (untrained base weights):

  * gathers layer-0's grouped experts.w1/w2/w3 via the REAL on-GPU
    ``_gather_tensor`` (= ``gather_dtensor_strided_safe`` over the
    ``(_StridedShard(fsdp), Shard(ep))`` composite), kept on CUDA, repeated 2x;
  * compares each expert row j to the BASE model's on-disk HF checkpoint
    (``safetensors.safe_open``) — a reference path that NEVER touches the EP gather;
  * emits the per-row corruption signature -> W1 (whole-expert swap m!=j) / W2
    (constant Δ row-block shift) / W3 (gather non-determinism) / W4 (dtype/byte) /
    CLEAN (=> corruption is DOWNSTREAM of the gather: NCCL broadcast or load_weights).

Why this is NON-circular (the fix vs the prior EXP2): the reference is the on-disk
base checkpoint, NOT a second traversal of the SAME gather. A gather that corrupts
identically on both sides can no longer look clean.

Run (4 nodes; DIAG_NUM_GPUS=16 total, 4 GPU/node). NO pytest needed (callable __main__):
    DIAG_NUM_GPUS=16 DIAG_GPUS_PER_NODE=4 python -m tests.gpu.diag_ep8_weight_disk_ref
"""
import os

import ray
import torch
from ray.util.placement_group import placement_group

from tests.gpu.utils import init_worker_with_type, get_test_actor_config, get_available_gpus
from skyrl_train.utils import initialize_ray, get_ray_pg_ready_with_timeout

# Geometry: EP=8 x FSDP=2 = 16 policy GPU. Laid out 4 nodes x 4 GPU/node so an
# 8-rank EP group (contiguous global ranks 0..7 — ep is the fastest mesh dim) spans
# node0(ranks0-3)+node1(ranks4-7) => CROSS-NODE. CP dropped (it is orthogonal to the
# weight gather; the prod composite under test is exactly (_StridedShard(fsdp),Shard(ep))).
MODEL = os.environ.get("DIAG_MODEL", "Qwen/Qwen3-30B-A3B-Thinking-2507")
DIAG_EP = int(os.environ.get("DIAG_EP", "8"))
DIAG_FSDP = int(os.environ.get("DIAG_FSDP", "2"))
NUM_GPUS = int(os.environ.get("DIAG_NUM_GPUS", str(DIAG_EP * DIAG_FSDP)))
GPUS_PER_NODE = int(os.environ.get("DIAG_GPUS_PER_NODE", "4"))
LAYER_IDX = int(os.environ.get("DIAG_LAYER", "0"))


def _get_cfg():
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.placement.colocate_all = False
    cfg.generator.model_dtype = "bfloat16"
    # Grouped-GEMM + EP=8 trainer (the prod (_StridedShard, Shard) composite).
    cfg.trainer.policy.fsdp_config.moe_grouped_gemm = True
    cfg.trainer.policy.fsdp_config.expert_model_parallel_size = DIAG_EP
    cfg.trainer.policy.fsdp_config.fsdp_size = DIAG_FSDP
    cfg.trainer.policy.fsdp_config.cpu_offload = True
    cfg.trainer.policy.fsdp_config.reshard_after_forward = True
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    cfg.trainer.policy.fsdp_config.ep_comm_backend = "torch"
    # 4 nodes x GPUS_PER_NODE.
    cfg.trainer.placement.policy_num_nodes = NUM_GPUS // GPUS_PER_NODE
    cfg.trainer.placement.policy_num_gpus_per_node = GPUS_PER_NODE
    # tiny micro batches; we never run a forward.
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    return cfg


def main():
    avail = get_available_gpus()
    print(f"[ep8diag] available GPUs on this node: {avail}", flush=True)

    pg = None
    try:
        cfg = _get_cfg()
        n_nodes = cfg.trainer.placement.policy_num_nodes
        print(f"[ep8diag] model={MODEL} EP={DIAG_EP} FSDP={DIAG_FSDP} "
              f"world={NUM_GPUS} layout={n_nodes}nodes x {GPUS_PER_NODE}gpu "
              f"| torch={torch.__version__}", flush=True)

        initialize_ray(cfg)

        # One bundle per GPU; PACK fills node-by-node so global rank N -> node N//gpus_per_node
        # (matches PPORayActorGroup's bundle_index = rank // num_gpus_per_node).
        pg = placement_group([{"GPU": 1, "CPU": 1}] * NUM_GPUS, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=600)

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=False,
            num_gpus_per_node=GPUS_PER_NODE,
            num_nodes=n_nodes,
            cfg=cfg,
        )

        # ---- 1. GEOMETRY PROOF: do EP groups straddle >=2 nodes? ----
        geos = ray.get(policy.async_run_ray_method("pass_through", "diag_ep8_geometry"))
        geos = sorted([g for g in geos if isinstance(g, dict)], key=lambda d: d["rank"])
        print("\n[ep8diag] ===== MESH GEOMETRY (rank | host | coord | ep_coord | ep_group_key) =====", flush=True)
        from collections import defaultdict
        group_hosts = defaultdict(set)
        for g in geos:
            print(f"    r{g['rank']:>2} host={g['host']:<24} coord={g['coord']} "
                  f"ep={g['ep_coord']} group={g['ep_group_key']} mesh={g['mesh_dim_names']}{g['mesh_shape']}",
                  flush=True)
            group_hosts[g["ep_group_key"]].add(g["host"])
        cross_node_groups = {k: hs for k, hs in group_hosts.items() if len(hs) >= 2}
        print(f"\n[ep8diag] EP groups: {len(group_hosts)} total; "
              f"{len(cross_node_groups)} span >=2 nodes.", flush=True)
        for k, hs in group_hosts.items():
            print(f"    group {k}: {len(hs)} node(s) -> {sorted(hs)}", flush=True)
        if not cross_node_groups:
            print("[ep8diag] !!! BLOCKER: NO EP group straddles 2 nodes. The geometry did NOT "
                  "force cross-node EP. Adjust DIAG_GPUS_PER_NODE / layout. ABORTING.", flush=True)
            return 3
        print(f"[ep8diag] PROOF OK: {len(cross_node_groups)}/{len(group_hosts)} EP groups span >=2 nodes "
              f"=> the EP=8 gather IS exercised cross-node.", flush=True)

        # ---- 2-5. NON-CIRCULAR on-GPU gather-vs-disk compare ----
        print(f"\n[ep8diag] ===== GATHER vs DISK-REFERENCE (layer {LAYER_IDX}, on-GPU) =====", flush=True)
        res = ray.get(policy.async_run_ray_method(
            "pass_through", "diag_ep8_disk_ref_compare", MODEL, LAYER_IDX, 2))
        r0 = next((d for d in res if isinstance(d, dict) and d.get("rank") == 0), None)
        if r0 is None:
            print("[ep8diag] !!! rank-0 returned no result.", flush=True)
            return 4
        print(f"[ep8diag] placements: {r0.get('placements')}", flush=True)
        print(f"[ep8diag] num_experts={r0.get('num_experts')} n_rep_gather={r0.get('n_rep_gather')}", flush=True)
        for line in r0["lines"]:
            print("    " + line, flush=True)
        if r0.get("wrong_expert_map"):
            print(f"\n[ep8diag] WRONG_EXPERT permutation map (gathered j -> disk m, m!=j):", flush=True)
            for j, m in sorted(r0["wrong_expert_map"].items()):
                print(f"    gathered w1[{j}]  ==  disk expert {m}", flush=True)

        print(f"\n[ep8diag] ===== VERDICT =====", flush=True)
        print(f"[ep8diag] {r0['verdict']}", flush=True)
        return 0
    finally:
        if pg is not None:
            try:
                ray.util.remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()


if __name__ == "__main__":
    import sys
    sys.exit(main())
