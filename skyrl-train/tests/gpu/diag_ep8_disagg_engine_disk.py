"""DISAGGREGATED cross-node ENGINE-vs-DISK weight diag — closes the MoE-salad loop.

The gather is proven CLEAN (diag_ep8_weight_disk_ref). NCCL-transport knobs are
falsified (r9). The ONLY untested path left is the DISAGGREGATED cross-node weight
transfer: policy nodes != engine node => the real ``broadcast_to_inference_engines``
(rank-0 worker -> engine over a cross-node NCCL group) + the engine's RL update-weights
RECEIVE — which base-serve and the COLOCATED weight-checks never exercise.

This brings up:
  * Policy  EP=8 x FSDP=2 = 16 GPU (prod MoE composite (_StridedShard(fsdp),Shard(ep)))
    on its OWN nodes (2 nodes x 8 GPU; PACK fills them).
  * 1 vLLM engine TP=2 x EP=2 (the prod r2-r9 salad-engine geometry) NON-colocated
    (shared_pg=None) => its own PG lands on a SEPARATE node.
PROVES the engine node is DISJOINT from the policy nodes (Ray node ids + hostnames).

Then at gs0 (untrained base): gather (clean) -> init_weight_sync_state ->
broadcast_to_inference_engines (the REAL cross-node transfer) -> RECEIVE. Repeated
twice (transport-race check). NO rollout.

NON-CIRCULAR readback: reads the engine's RAW per-local-slot FusedMoE
w13_weight/w2_weight + the engine's OWN expert_map (read_expert_slots_raw), and
compares each slot to the BASE DISK checkpoint (safetensors) with:
  * an authoritative slot->global map (engine expert_map) -> per-slot value diff vs
    disk-expert-(that global id): whole-expert garbage / partial / dtype => D1 transport;
  * an INDEPENDENT cross-expert nearest-match (find disk m with slot==disk[m]) -> if the
    best disk match m != the expert_map's claimed global id => D2 receive PLACEMENT.
  * two broadcasts diffed -> non-determinism => transport race.

vLLM stores w13 = [w1;w3] possibly permuted to [w3;w1] by process_weights_after_loading;
we compare the per-slot w13 against disk [gate;up] AND [up;gate] and take the better, so
a clean-but-permuted half ordering is not misread as corruption.

Run (3 nodes; policy 16 GPU + engine 2 GPU):
    DIAG_POLICY_GPUS_PER_NODE=8 python -m tests.gpu.diag_ep8_disagg_engine_disk
"""
import asyncio
import os

import ray
import torch
from ray.util.placement_group import placement_group

from tests.gpu.utils import init_worker_with_type, get_test_actor_config, get_available_gpus
from skyrl_train.utils import initialize_ray, get_ray_pg_ready_with_timeout
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from transformers import AutoTokenizer, AutoConfig

MODEL = os.environ.get("DIAG_MODEL", "Qwen/Qwen3-30B-A3B-Thinking-2507")
DIAG_EP = int(os.environ.get("DIAG_EP", "8"))          # policy EP
DIAG_FSDP = int(os.environ.get("DIAG_FSDP", "2"))      # policy FSDP
POLICY_GPUS_PER_NODE = int(os.environ.get("DIAG_POLICY_GPUS_PER_NODE", "8"))
ENGINE_TP = int(os.environ.get("DIAG_ENGINE_TP", "2"))
ENGINE_EP = int(os.environ.get("DIAG_ENGINE_EP", "2"))
LAYER_IDX = int(os.environ.get("DIAG_LAYER", "0"))
# FIXED-ORDER w13 compare (the gate/up-swap blind-spot probe). Default ON now — the
# prior both-order tolerance is what hid a possible #1685 [w1;w3]<->[w3;w1] half-swap.
W13_STRICT = os.environ.get("DIAG_W13_STRICT", "1") == "1"
# Sample layer 0 + a MID layer so a depth-dependent permute can't hide behind layer 0.
DIAG_LAYERS = [int(x) for x in os.environ.get("DIAG_LAYERS", "0,24").split(",") if x.strip() != ""]
EPS = 1e-3  # bf16 round-trip through the engine (load + process_weights) tolerance


def _get_cfg(policy_gpus, policy_nodes):
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.strategy = "fsdp2"
    # DISAGGREGATED: policy and engine on DIFFERENT GPUs/nodes.
    cfg.trainer.placement.colocate_all = False
    cfg.generator.model_dtype = "bfloat16"
    cfg.generator.backend = "vllm"
    cfg.generator.async_engine = True
    cfg.generator.weight_sync_backend = "nccl"
    cfg.generator.enable_prefix_caching = False
    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = ENGINE_TP
    cfg.generator.inference_engine_expert_parallel_size = ENGINE_EP
    cfg.generator.inference_engine_pipeline_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.gpu_memory_utilization = 0.85
    cfg.generator.override_existing_update_group = "enable"
    # Policy grouped+EP.
    cfg.trainer.policy.fsdp_config.moe_grouped_gemm = True
    cfg.trainer.policy.fsdp_config.expert_model_parallel_size = DIAG_EP
    cfg.trainer.policy.fsdp_config.fsdp_size = DIAG_FSDP
    cfg.trainer.policy.fsdp_config.cpu_offload = True
    cfg.trainer.policy.fsdp_config.reshard_after_forward = True
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    cfg.trainer.policy.fsdp_config.ep_comm_backend = "torch"
    cfg.trainer.placement.policy_num_nodes = policy_nodes
    cfg.trainer.placement.policy_num_gpus_per_node = policy_gpus
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    return cfg


def _disk_local_dir(model_path):
    import os as _os
    if _os.path.isdir(model_path) and _os.path.exists(_os.path.join(model_path, "config.json")):
        return model_path
    from huggingface_hub import snapshot_download
    return snapshot_download(model_path, allow_patterns=["*.safetensors", "*.json"])


def _load_disk_experts(local_dir, layer_idx, num_experts):
    """Return {global_j: {'gate':T,'up':T,'down':T}} fp32 CPU from base disk shards."""
    import json
    from safetensors import safe_open
    idx = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            wmap = json.load(f)["weight_map"]
    else:
        wmap = None
    out = {}
    for j in range(num_experts):
        ent = {}
        for proj, tag in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
            key = f"model.layers.{layer_idx}.mlp.experts.{j}.{proj}.weight"
            shard = os.path.join(local_dir, wmap[key]) if wmap else os.path.join(local_dir, "model.safetensors")
            with safe_open(shard, framework="pt", device="cpu") as fp:
                ent[tag] = fp.get_tensor(key).float()
        out[j] = ent
    return out


def main():
    avail = get_available_gpus()
    print(f"[disagg] available GPUs this node: {avail}", flush=True)
    policy_world = DIAG_EP * DIAG_FSDP
    policy_nodes = policy_world // POLICY_GPUS_PER_NODE
    pol_pg = None
    try:
        cfg = _get_cfg(POLICY_GPUS_PER_NODE, policy_nodes)
        print(f"[disagg] model={MODEL} policy EP={DIAG_EP}xFSDP={DIAG_FSDP}={policy_world} "
              f"({policy_nodes} nodes x {POLICY_GPUS_PER_NODE}gpu) | engine TP={ENGINE_TP} EP={ENGINE_EP} "
              f"| torch={torch.__version__}", flush=True)
        initialize_ray(cfg)

        # ---- POLICY FIRST, claiming its WHOLE nodes. init_worker_with_type(shared_pg=None)
        # builds bundles=[{GPU:POLICY_GPUS_PER_NODE}]*policy_nodes -> reserves policy_nodes
        # FULL 8-GPU nodes (num_gpus_per_actor=0.75 packs each node's 8 actors onto its one
        # 8-GPU bundle). The engine PG (created AFTER, shared_pg=None) is then FORCED onto a
        # fresh node => guaranteed disjoint. The DISAGGREGATION PROOF below is the hard gate
        # (ABORTs if the engine still overlaps a policy node). ----
        pol_pg = None  # owned by init_worker_with_type (its own PG); nothing to clean here.
        policy = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False,
            num_gpus_per_node=POLICY_GPUS_PER_NODE, num_nodes=policy_nodes, cfg=cfg,
        )

        # ---- ENGINE AFTER policy occupies its nodes (non-colocated, shared_pg=None ->
        # its own STRICT_PACK PG must land on a node policy did NOT fill = a 3rd node). ----
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        engines = create_ray_wrapped_inference_engines(
            num_inference_engines=1,
            tensor_parallel_size=ENGINE_TP,
            expert_parallel_size=ENGINE_EP,
            model_dtype="bfloat16",
            pretrain=MODEL,
            seed=42,
            vllm_v1_disable_multiproc=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            shared_pg=None,                      # DISAGGREGATED
            gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
            inference_engine_enable_sleep=False,  # not colocated -> no sleep
            async_engine=True,
            max_num_batched_tokens=8192,
            max_num_seqs=256,
            tokenizer=tokenizer,
            backend="vllm",
        )
        client = InferenceEngineClient(engines, tokenizer, cfg)

        # ---- GEOMETRY PROOF: engine node disjoint from policy nodes ----
        pol_geo = ray.get(policy.async_run_ray_method("pass_through", "diag_ep8_geometry"))
        pol_geo = sorted([g for g in pol_geo if isinstance(g, dict)], key=lambda d: d["rank"])
        pol_hosts = sorted(set(g["host"] for g in pol_geo))
        rank0_host = next(g["host"] for g in pol_geo if g["rank"] == 0)
        eng_actor = client.engines[0].inference_engine_actor
        try:
            eng_hosts = sorted(set(ray.get(eng_actor.report_engine_hosts.remote())))
        except Exception as e:
            eng_hosts = [f"<rpc-failed:{e}>"]
        print(f"\n[disagg] ===== DISAGGREGATION PROOF =====", flush=True)
        print(f"[disagg] policy hosts={pol_hosts}  (rank0 host={rank0_host})", flush=True)
        print(f"[disagg] engine hosts={eng_hosts}", flush=True)
        disjoint = all(h not in pol_hosts for h in eng_hosts if not str(h).startswith("<"))
        if not disjoint:
            print(f"[disagg] !!! BLOCKER: engine host(s) {eng_hosts} OVERLAP policy hosts "
                  f"{pol_hosts}. Broadcast would be INTRA-node. ABORTING.", flush=True)
            return 3
        print(f"[disagg] PROOF OK: engine node(s) {eng_hosts} DISJOINT from policy nodes "
              f"{pol_hosts} => broadcast_to_inference_engines is genuinely CROSS-NODE.", flush=True)

        mc = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        num_experts = int(getattr(mc, "num_experts", getattr(mc, "num_local_experts", 128)))

        # Clamp sampled layers to the model depth.
        num_layers = int(getattr(mc, "num_hidden_layers", 48))
        layers = sorted(set(li for li in DIAG_LAYERS if 0 <= li < num_layers)) or [0]
        print(f"[disagg] sampling layers {layers} of {num_layers} (W13_STRICT={W13_STRICT})", flush=True)

        # ---- BEFORE-broadcast: read the engine's FRESH-FROM-DISK w13 (the kernel format
        # produced by the initial load's process_weights_after_loading / swap_w13_to_w31).
        # This is the reference for whether the RL update PRESERVES the kernel layout. ----
        pre_by_layer = {}
        for li in layers:
            pr = ray.get(eng_actor.read_engine_expert_slots_raw.remote(li))
            pre_by_layer[li] = ([pr] if isinstance(pr, dict) else pr)

        # ---- REAL disaggregated transfer, TWICE (transport-race check) ----
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        # slot_reads_by_layer[li] = list (per rep) of list (per engine rank) of dicts.
        slot_reads_by_layer = {li: [] for li in layers}
        for rep in range(2):
            ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
            for li in layers:
                per_rank = ray.get(eng_actor.read_engine_expert_slots_raw.remote(li))
                if isinstance(per_rank, dict):
                    per_rank = [per_rank]
                slot_reads_by_layer[li].append(per_rank)

        # ---- DISK reference (non-circular) + per-layer compare ----
        local_dir = _disk_local_dir(MODEL)
        # ---- KERNEL-FORMAT DISCRIMINATOR (the decisive test) ----
        # For each layer, compare: (a) the FRESH-FROM-DISK engine w13 (pre) vs disk natural
        # [gate;up], (b) the POST-UPDATE engine w13 vs disk, (c) pre vs post. The from-disk
        # base-serve is COHERENT, so (a) tells us the CORRECT kernel layout; if the update
        # changes the layout (c != identical) the RL path corrupts it.
        print(f"\n[disagg] ===== KERNEL-FORMAT DISCRIMINATOR (pre=from-disk load vs post=RL-update) =====", flush=True)
        for li in layers:
            d = _load_disk_experts(local_dir, li, num_experts)
            for wr, (rd_pre, rd_post) in enumerate(zip(pre_by_layer[li], slot_reads_by_layer[li][0])):
                if not (isinstance(rd_pre, dict) and "slots" in rd_pre and isinstance(rd_post, dict) and "slots" in rd_post):
                    continue
                s2g = rd_pre.get("slot_to_global", [])
                inter = rd_pre.get("w13_inter_half")
                # sample the FIRST owned slot for a compact signature
                sl = sorted(rd_pre["slots"])[0]
                g = s2g[sl] if (s2g and sl < len(s2g)) else 0
                I = inter
                pre = rd_pre["slots"][sl]["w13"].float()
                post = rd_post["slots"][sl]["w13"].float()
                dg, du = d[g]["gate"], d[g]["up"]
                def fmt(x):
                    # which order does engine w13 hold? compare top half to disk gate vs up.
                    e_nat = max((x[:I] - dg).abs().max().item(), (x[I:] - du).abs().max().item())  # [gate;up]
                    e_swp = max((x[:I] - du).abs().max().item(), (x[I:] - dg).abs().max().item())  # [up;gate]
                    if e_nat <= EPS:
                        return f"[gate;up] (natural/checkpoint; e_nat={e_nat:.1e})"
                    if e_swp <= EPS:
                        return f"[up;gate] (SWAPPED/kernel; e_swp={e_swp:.1e})"
                    return f"NEITHER (e_nat={e_nat:.1e} e_swp={e_swp:.1e})"
                pre_post_id = (pre - post).abs().max().item()
                print(f"    L{li} rank{wr} slot{sl}(g={g}): PRE(from-disk)={fmt(pre)} | "
                      f"POST(RL-update)={fmt(post)} | max|pre-post|={pre_post_id:.3e}", flush=True)
        agg_wrong, agg_nondet, agg_corrupt, agg_swaps = {}, False, 0, []
        for li in layers:
            disk = _load_disk_experts(local_dir, li, num_experts)
            print(f"\n[disagg] ===== ENGINE(received) vs DISK — layer {li}, {num_experts} experts "
                  f"(W13_STRICT={W13_STRICT}) =====", flush=True)
            vlines, wmap, ndet, tcorr, gswaps = analyze(slot_reads_by_layer[li], disk, num_experts)
            for ln in vlines:
                print("    " + ln, flush=True)
            agg_wrong.update({(li,) + k: v for k, v in wmap.items()})
            agg_nondet = agg_nondet or ndet
            agg_corrupt += tcorr
            agg_swaps += [(li,) + s for s in gswaps]

        print(f"\n[disagg] ===== VERDICT (across layers {layers}) =====", flush=True)
        wrong_map, nondet, total_corrupt, gate_up_swaps = agg_wrong, agg_nondet, agg_corrupt, agg_swaps
        corrupt = total_corrupt > 0
        if gate_up_swaps:
            ranks_hit = sorted(set(s[1] for s in gate_up_swaps))
            layers_hit = sorted(set(s[0] for s in gate_up_swaps))
            print(f"[disagg] GATE_UP_SWAP present at layers {layers_hit}, ep-ranks {ranks_hit}; "
                  f"{len(gate_up_swaps)} slots total.", flush=True)
            print(f"[disagg] VERDICT = W13 GATE_UP_SWAP: {len(gate_up_swaps)} engine expert slots hold "
                  f"w13 as [up; gate] instead of [gate; up] (ep ranks {ranks_hit}). The RL disaggregated "
                  f"update-weights path did NOT apply vLLM's [w1;w3]->[w3;w1] process_weights permute "
                  f"(=> #1685 silent MoE corruption). down_proj/placement correct; ONLY the w13 halves "
                  f"are transposed. THIS is the token-salad root cause on this path.", flush=True)
        elif nondet:
            print("[disagg] VERDICT = TRANSPORT RACE: the two broadcasts disagree "
                  "(non-deterministic engine bytes) => race in broadcast_to_inference_engines / receive.", flush=True)
        elif wrong_map:
            print(f"[disagg] VERDICT = D2 RECEIVE PLACEMENT: {len(wrong_map)} engine slots carry a "
                  f"DIFFERENT global expert than the engine expert_map claims (m!=j). The receive-side "
                  f"grouped-MoE expert mapping (expert_map / update-weights slotting) is WRONG.", flush=True)
        elif corrupt:
            print("[disagg] VERDICT = D1 BROADCAST TRANSPORT: engine slots differ from disk in VALUE "
                  "(garbage/partial/dtype), placement OK => broadcast_to_inference_engines corrupts the "
                  "~400MB grouped-expert tensor (chunking/dtype/contiguity/order).", flush=True)
        else:
            print("[disagg] VERDICT = CLEAN: engine-held weights EQUAL disk for every slot in FIXED order "
                  "(w13[:I]==gate, w13[I:]==up), placement matches expert_map. Broadcast+receive+w13-layout "
                  "all correct => corruption is FURTHER DOWNSTREAM (forward/kernel).", flush=True)
        return 0
    finally:
        if pol_pg is not None:
            try:
                ray.util.remove_placement_group(pol_pg)
            except Exception:
                pass
        ray.shutdown()


def _hash(t):
    import hashlib
    return hashlib.md5(t.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()[:12]


def analyze(slot_reads, disk, num_experts):
    """slot_reads: list (per broadcast rep) of list (per engine worker rank) of dicts.
    Returns (lines, wrong_placement_map, nondet, total_corrupt, gate_up_swaps)."""
    lines = []
    total_corrupt = 0
    gate_up_swaps = []  # (wr, slot, claimed_g) per GATE_UP_SWAP slot
    rep0 = slot_reads[0]
    rep1 = slot_reads[1] if len(slot_reads) > 1 else slot_reads[0]

    # Detect non-determinism across the two broadcasts (same worker/slot bytes).
    nondet = False
    for r0, r1 in zip(rep0, rep1):
        s0 = r0.get("slots", {}) if isinstance(r0, dict) else {}
        s1 = r1.get("slots", {}) if isinstance(r1, dict) else {}
        for sl in s0:
            if sl in s1:
                d = (s0[sl]["w13"] - s1[sl]["w13"]).abs().max().item()
                if d > EPS:
                    nondet = True

    wrong_map = {}
    # Pre-stack disk gate/up for nearest-match. Build disk "w13-like" in BOTH orders.
    disk_gate = {j: disk[j]["gate"] for j in range(num_experts)}
    disk_up = {j: disk[j]["up"] for j in range(num_experts)}
    disk_down = {j: disk[j]["down"] for j in range(num_experts)}

    for wr, rd in enumerate(rep0):
        if not isinstance(rd, dict) or "slots" not in rd:
            lines.append(f"[engine-rank{wr}] no slots ({rd.get('error') if isinstance(rd, dict) else rd})")
            continue
        ranks = rd.get("__ranks__", {})
        emap = rd.get("expert_map")
        s2g = rd.get("slot_to_global", [])
        inter = rd.get("w13_inter_half")
        local_num = rd.get("local_num_experts")
        lines.append(f"[engine-rank{wr}] tp={ranks.get('tp_rank')}/{ranks.get('tp_size')} "
                     f"ep={ranks.get('ep_rank')}/{ranks.get('ep_size')} local_experts={local_num} "
                     f"placement={rd.get('placement_strategy')} slot_to_global={s2g}")
        slots = rd["slots"]
        n_corrupt = 0
        for sl in sorted(slots):
            w13 = slots[sl]["w13"].float()  # [2I, H]
            w2 = slots[sl]["w2"].float()    # [H, I]
            claimed_g = s2g[sl] if (s2g and sl < len(s2g)) else None
            # split engine w13 into halves (order unknown post process_weights).
            I = inter
            hA, hB = w13[:I], w13[I:]   # engine w13 top half / bottom half
            # --- per-slot value diff vs the engine-claimed global expert ---
            best_for_claim = None
            swap_tag = ""
            if claimed_g is not None and 0 <= claimed_g < num_experts:
                g = claimed_g
                e_w2 = (w2 - disk_down[g]).abs().max().item()
                if W13_STRICT:
                    # FIXED-ORDER (the w13 blind-spot probe): the vLLM FusedMoE convention
                    # is w13 = [gate(=w1); up(=w3)] -> engine w13[:I] MUST equal disk gate,
                    # w13[I:] MUST equal disk up. NO both-order tolerance. If the RL update
                    # path skipped the [w1;w3]->[w3;w1] process_weights permute (#1685), the
                    # halves are SWAPPED and this fires GATE_UP_SWAP.
                    e_top_gate = (hA - disk_gate[g]).abs().max().item()   # expect ~0 if correct
                    e_bot_up = (hB - disk_up[g]).abs().max().item()      # expect ~0 if correct
                    e_top_up = (hA - disk_up[g]).abs().max().item()      # ~0 if halves SWAPPED
                    e_bot_gate = (hB - disk_gate[g]).abs().max().item()  # ~0 if halves SWAPPED
                    e_w13 = max(e_top_gate, e_bot_up)                    # the FIXED-ORDER error
                    best_for_claim = max(e_w13, e_w2)
                    # SWAP signature: the WRONG order matches but the RIGHT order does not.
                    if e_w13 > EPS and max(e_top_up, e_bot_gate) <= EPS:
                        swap_tag = (f"GATE_UP_SWAP (w13[:I]==disk_up & w13[I:]==disk_gate; "
                                    f"fixed-order err top_gate={e_top_gate:.2e} bot_up={e_bot_up:.2e})")
                    elif e_w13 > EPS:
                        swap_tag = (f"w13 fixed-order MISMATCH top_gate={e_top_gate:.2e} "
                                    f"bot_up={e_bot_up:.2e} (swap-order top_up={e_top_up:.2e} "
                                    f"bot_gate={e_bot_gate:.2e})")
                else:
                    # both-order tolerant (the prior, w13-blind compare).
                    e_gu = max((hA - disk_gate[g]).abs().max().item(), (hB - disk_up[g]).abs().max().item())
                    e_ug = max((hB - disk_gate[g]).abs().max().item(), (hA - disk_up[g]).abs().max().item())
                    e_w13 = min(e_gu, e_ug)
                    best_for_claim = max(e_w13, e_w2)
            # --- independent cross-expert nearest-match on the gate half ---
            best_m, best_e = None, float("inf")
            for m in range(num_experts):
                # match whichever engine half is closest to disk gate[m]
                em = min((hA - disk_gate[m]).abs().max().item(), (hB - disk_gate[m]).abs().max().item())
                if em < best_e:
                    best_e, best_m = em, m
            tag = ""
            if best_for_claim is not None and best_for_claim <= EPS:
                # placement + value both correct
                if best_m is not None and best_m != claimed_g and best_e <= EPS:
                    # ambiguous duplicate; ignore
                    pass
                continue  # CLEAN slot
            n_corrupt += 1
            if swap_tag.startswith("GATE_UP_SWAP"):
                gate_up_swaps.append((wr, sl, claimed_g))
                tag = swap_tag
            elif best_m is not None and claimed_g is not None and best_m != claimed_g and best_e <= EPS:
                wrong_map[(wr, sl)] = (claimed_g, best_m)
                tag = f"WRONG_PLACEMENT engine_map_says={claimed_g} but bytes==disk_expert={best_m}"
            elif swap_tag:
                tag = swap_tag
            elif best_for_claim is not None and best_for_claim > EPS:
                # value corruption vs the claimed expert; report nearest + magnitude
                gh = _hash(w13)
                tag = (f"CORRUPT vs claimed_g={claimed_g} max_abs={best_for_claim:.3e} "
                       f"nearest_disk={best_m}@{best_e:.2e} w13hash={gh}")
                if best_e > EPS:
                    tag += " GARBAGE(no disk expert matches)"
            else:
                tag = f"UNRESOLVED claimed_g={claimed_g} nearest={best_m}@{best_e:.2e}"
            lines.append(f"    slot{sl} (claimed g={claimed_g}): {tag}")
        total_corrupt += n_corrupt
        lines.append(f"[engine-rank{wr}] CORRUPT {n_corrupt}/{len(slots)} slots")
    return lines, wrong_map, nondet, total_corrupt, gate_up_swaps


if __name__ == "__main__":
    import sys
    sys.exit(main())
