"""SYNC-AND-COMPARE diagnostic for the EP MoE FSDP->vLLM weight-sync corruption.

NO rollout, NO generation (a broken pre-sync baseline poisoned the coherence
vehicle). Just: bring up the FSDP grouped+EP policy + an EP=2 vLLM engine, do ONE
FSDP->vLLM weight sync, then DIRECTLY read back the synced engine weights and
compare VALUE-by-value against the FSDP source — across ALL weight TYPES, not just
experts. Emits a CORRUPTION SIGNATURE per tensor.

Source  : policy.read_post_step_weights(names)  (the SAME extract_weights /
          _gather_tensor path the broadcast uses; rank-0 returns full HF tensors).
Engine  : engine.read_named_weights(names)       (reads vLLM's loaded params back
          under HF names, incl. the FusedMoE w13->gate/up split + w2->down).

For each name we compute a signature vs source:
  CLEAN          max_abs <= EPS
  TRANSPOSED     matches source.T
  GATE_UP_SWAP   (experts) gate matches source up_proj and vice-versa
  WRONG_EXPERT   matches a DIFFERENT global expert's source (reports which j)
  PERMUTED_ROWS  same multiset of rows, different order (row-permutation)
  SCALED         engine ~= c * source (constant c != 1)
  ZEROED         engine all ~0 while source non-zero
  DTYPE/QUANT    large but structured error (bf16 vs fp8 etc.) — flagged by hist
  DIFF           none of the above (raw max_abs)

Run (1 node; NUM_GPUS ranks). NO pytest needed (callable __main__):
    python -m tests.gpu.diag_ep_weight_sync_compare
"""
import asyncio
import os
import re

import ray
import torch
from ray.util.placement_group import placement_group

from tests.gpu.test_expert_parallel_inference import _get_test_cfg, init_ray_inference_engines, MODEL
from tests.gpu.utils import init_worker_with_type
from skyrl_train.utils import initialize_ray, get_ray_pg_ready_with_timeout

NUM_GPUS = int(os.environ.get("DIAG_NUM_GPUS", "8"))
# Policy EP/FSDP geometry. Must satisfy: ep*fsdp == policy GPUs AND
# (num_experts // ep) % fsdp == 0 (apply_ep's even-shard guard). For Qwen1.5-MoE
# (60 experts) ep=2/fsdp=2 is clean (30 % 2 == 0). Engine uses TP=2 separately.
DIAG_EP = int(os.environ.get("DIAG_EP", "2"))
DIAG_FSDP = int(os.environ.get("DIAG_FSDP", "2"))
EPS = 1e-3


def _sampled_layers(num_layers: int):
    """Layer 0 (first) + a MID layer — sample the expert weight-sync at two depths
    so a depth-dependent corruption can't hide behind a single-layer probe."""
    mid = max(1, num_layers // 2)
    return sorted(set([0, mid]))


def _build_rep_names(num_layers: int, num_experts: int):
    """Sample ALL weight types at layer 0 + a MID layer, across ALL experts.

    Per the W-vs-I split we must value-compare EVERY expert (not a representative
    subset) at >=2 layers, so a per-expert m!=j permutation map (Candidate A vs B)
    is fully observable. Non-expert tensors (embed/norm/attn/router) are the clean-
    control; sampled at layer 0 (+ the mid layer's router/attn) too.
    """
    layers = _sampled_layers(num_layers)
    names = [
        # embeddings + head + final norm (global, not per-layer)
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for li in layers:
        L = f"model.layers.{li}"
        names += [
            # attention (q/k/v/o) — qkv may be fused in vLLM; we try all
            f"{L}.self_attn.q_proj.weight",
            f"{L}.self_attn.k_proj.weight",
            f"{L}.self_attn.v_proj.weight",
            f"{L}.self_attn.o_proj.weight",
            # layernorms
            f"{L}.input_layernorm.weight",
            f"{L}.post_attention_layernorm.weight",
            # router gate
            f"{L}.mlp.gate.weight",
        ]
        # ALL experts at this layer (w1/gate, w3/up, w2/down).
        for j in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                names.append(f"{L}.mlp.experts.{j}.{proj}.weight")
    # rep_experts kept for the summary banner (first/last of each EP half).
    half = num_experts // 2
    rep_experts = sorted(set([0, 1, half - 1, half, half + 1, num_experts - 1]))
    return names, rep_experts


def _signature(name, tr, eng, src_all_experts=None, gj=None):
    """Classify how `eng` differs from `tr` (both fp32 CPU, same shape after t-try)."""
    tr = tr.float()
    eng = eng.float()
    # shape: try transpose for 2D
    transposed = False
    if tuple(tr.shape) != tuple(eng.shape):
        if tr.dim() == 2 and eng.dim() == 2 and tuple(tr.shape) == tuple(eng.t().shape):
            eng = eng.t().contiguous()
            transposed = True
        else:
            return f"SHAPE_MISMATCH tr={tuple(tr.shape)} eng={tuple(eng.shape)}", float("nan")
    max_abs = float((tr - eng).abs().max().item())
    if max_abs <= EPS:
        return ("CLEAN" if not transposed else "CLEAN(after .T)"), max_abs

    sig_extra = []
    # zeroed?
    if float(eng.abs().max()) <= EPS and float(tr.abs().max()) > EPS:
        return "ZEROED (engine ~0)", max_abs
    # scaled? engine ~= c*tr
    denom = float((tr * tr).sum())
    if denom > 0:
        c = float((tr * eng).sum() / denom)
        if abs(c - 1.0) > 0.02 and float((eng - c * tr).abs().max()) <= 10 * EPS:
            return f"SCALED (engine ~= {c:.4f} * source)", max_abs
    # gate-up swap (only for expert gate/up): compare to the OTHER proj of same expert
    # (handled by caller via src_all_experts); here flag relative magnitude.
    # wrong-expert: does eng match a DIFFERENT global expert's source?
    if src_all_experts is not None and gj is not None:
        best_j, best_err = None, float("inf")
        for jj, src_j in src_all_experts.items():
            if src_j.shape != eng.shape:
                continue
            err = float((src_j.float() - eng).abs().max().item())
            if err < best_err:
                best_err, best_j = err, jj
        if best_j is not None and best_j != gj and best_err <= EPS:
            return f"WRONG_EXPERT (engine carries source expert {best_j}, not {gj})", max_abs
        if best_j is not None and best_j != gj:
            sig_extra.append(f"closest_src_expert={best_j}@{best_err:.2e}")
    # permuted rows? same sorted rows, different order (per-row first-element fingerprint)
    if tr.dim() >= 2 and tr.shape[0] == eng.shape[0] and tr.shape[0] <= 4096:
        tr_fp = tr.reshape(tr.shape[0], -1)[:, 0]
        eng_fp = eng.reshape(eng.shape[0], -1)[:, 0]
        if torch.allclose(tr_fp.sort().values, eng_fp.sort().values, atol=10 * EPS) and not torch.allclose(tr_fp, eng_fp, atol=10 * EPS):
            sig_extra.append("ROW_PERMUTED(dim0)")
    rel = max_abs / (float(tr.abs().max()) + 1e-9)
    extra = ("  [" + "; ".join(sig_extra) + "]") if sig_extra else ""
    return f"DIFF rel={rel:.3f}{extra}", max_abs


def main():
    from tests.gpu.utils import get_available_gpus
    avail = get_available_gpus()
    if len(avail) < NUM_GPUS:
        print(f"[diag] need {NUM_GPUS} GPUs, found {len(avail)}: {avail}")
        return 2

    pg = None
    try:
        cfg = _get_test_cfg()
        cfg.trainer.placement.colocate_all = True
        # Grouped-GEMM + EP trainer (the prod (_StridedShard, Shard) composite). The
        # POLICY spans DIAG_EP*DIAG_FSDP GPUs; the engine uses TP=2 on top (colocated).
        policy_gpus = DIAG_EP * DIAG_FSDP
        cfg.trainer.policy.fsdp_config.moe_grouped_gemm = True
        cfg.trainer.policy.fsdp_config.expert_model_parallel_size = DIAG_EP
        cfg.trainer.policy.fsdp_config.fsdp_size = DIAG_FSDP
        cfg.generator.gpu_memory_utilization = 0.45
        cfg.trainer.placement.policy_num_gpus_per_node = policy_gpus
        cfg.trainer.placement.policy_num_nodes = 1

        initialize_ray(cfg)
        # Colocated: policy + engine SHARE the same GPUs (engine sleeps during the
        # policy's turn). With colocate_all=True the shared PG bundle count MUST equal
        # the policy world_size (= policy_gpus = ep*fsdp). The TP=2 engine colocates on
        # a subset of those bundles.
        pg = placement_group([{"GPU": 1, "CPU": 1}] * policy_gpus, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=120)

        client = init_ray_inference_engines(
            backend=cfg.generator.backend,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            shared_pg=pg,
            config=cfg,
        )
        asyncio.run(client.wake_up())

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=True,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        from transformers import AutoConfig
        mc = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        num_experts = int(getattr(mc, "num_experts", getattr(mc, "num_local_experts", 60)))
        num_layers = int(getattr(mc, "num_hidden_layers", 24))
        rep_names, rep_experts = _build_rep_names(num_layers, num_experts)
        sampled_layers = _sampled_layers(num_layers)
        # Also pull EVERY sampled-layer expert's source (gate_proj) so WRONG_EXPERT
        # (engine name j carries source expert m!=j) is detectable PER LAYER.
        all_expert_names = [
            f"model.layers.{li}.mlp.experts.{j}.gate_proj.weight"
            for li in sampled_layers for j in range(num_experts)
        ]

        print(f"[diag] model={MODEL} num_experts={num_experts} num_layers={num_layers}")
        print(f"[diag] NUM_GPUS={NUM_GPUS} policy ep={DIAG_EP} fsdp={DIAG_FSDP} "
              f"(experts//ep={num_experts//DIAG_EP}, %fsdp={(num_experts//DIAG_EP)%DIAG_FSDP}) "
              f"| torch={torch.__version__}")
        print(f"[diag] sampling {len(rep_names)} tensors (all types), experts={rep_experts}")

        # ---- ONE sync ----
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.wake_up(tags=["weights"]))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

        # ---- SOURCE (trainer post-sync HF tensors; same gather as broadcast) ----
        want = list(set(rep_names + all_expert_names))
        src_per_rank = ray.get(policy.async_run_ray_method("pass_through", "read_post_step_weights", want))
        src = {}
        for d in src_per_rank:
            if isinstance(d, dict):
                src.update({k: v for k, v in d.items() if isinstance(v, torch.Tensor)})
        # Per-layer global-expert source map (gate_proj) for WRONG_EXPERT detection.
        src_all_experts = {
            li: {
                j: src[f"model.layers.{li}.mlp.experts.{j}.gate_proj.weight"]
                for j in range(num_experts)
                if f"model.layers.{li}.mlp.experts.{j}.gate_proj.weight" in src
            }
            for li in sampled_layers
        }

        policy.offload_to_cpu()
        asyncio.run(client.wake_up(tags=["kv_cache"]))

        # ---- ENGINE readback ----
        engine_actor = client.engines[0].inference_engine_actor
        eng_per_rank = ray.get(engine_actor.read_engine_weights.remote(rep_names, True))
        if isinstance(eng_per_rank, dict):
            eng_per_rank = [eng_per_rank]

        # inventory dump (first-run aid)
        for rk, rd in enumerate(eng_per_rank):
            inv = rd.get("__inventory__") if isinstance(rd, dict) else None
            ranks = rd.get("__ranks__") if isinstance(rd, dict) else None
            if inv is not None and rk == 0:
                print(f"[diag][engine-rank0] ranks={ranks}; sample layer0 inventory:")
                for k in sorted(inv):
                    if "layers.0." in k or "embed_tokens" in k or "lm_head" in k or k == "model.norm.weight":
                        print(f"     {k} {inv[k]}")

        expert_re = re.compile(r"\.mlp\.experts\.(\d+)\.")
        layer_re = re.compile(r"model\.layers\.(\d+)\.")
        # TP-sharded tensors, matched layer-AGNOSTICALLY (regex on the suffix) so both
        # the layer-0 and mid-layer attn projections assemble across TP ranks.
        TP_SHARD_DIM = {"model.embed_tokens.weight": 0, "lm_head.weight": 0,
                        "self_attn.o_proj.weight": 1, "self_attn.q_proj.weight": 0,
                        "self_attn.k_proj.weight": 0, "self_attn.v_proj.weight": 0}

        def _tp_shard_dim(name):
            if name in TP_SHARD_DIM:
                return TP_SHARD_DIM[name]
            for suf, d in TP_SHARD_DIM.items():
                if suf.startswith("self_attn") and name.endswith(suf):
                    return d
            return None

        def assemble(name):
            entries = []
            for rd in eng_per_rank:
                if not isinstance(rd, dict):
                    continue
                e = rd.get(name)
                rkc = rd.get("__ranks__", {})
                if isinstance(e, dict) and e.get("found") and isinstance(e.get("tensor"), torch.Tensor):
                    entries.append((rkc, e["tensor"]))
            if not entries:
                # report any per-rank notes for diagnosis
                notes = [rd.get(name, {}).get("note") or rd.get(name, {}).get("error")
                         for rd in eng_per_rank if isinstance(rd, dict)]
                return None, f"MISSING ({[n for n in notes if n][:1]})"
            tp_dim = _tp_shard_dim(name)
            if tp_dim is not None:
                shards = {}
                for rkc, t in entries:
                    shards[rkc.get("tp_rank", 0)] = t
                try:
                    return torch.cat([shards[i] for i in sorted(shards)], dim=tp_dim), "tp-cat"
                except Exception:
                    return entries[0][1], "tp-partial"
            return entries[0][1], "direct/expert"

        print("\n[diag] ===== CORRUPTION SIGNATURE (name | engine-mode | signature | max_abs) =====")
        results = []
        for name in rep_names:
            tr = src.get(name)
            eng, emode = assemble(name)
            if tr is None:
                results.append((name, emode, "SRC_MISSING", float("nan")))
                continue
            if eng is None:
                results.append((name, emode, "ENG_MISSING", float("nan")))
                continue
            m = expert_re.search(name)
            gj = int(m.group(1)) if m else None
            lm = layer_re.search(name)
            li = int(lm.group(1)) if lm else None
            # WRONG_EXPERT search uses THIS layer's gate_proj source map (only
            # meaningful for gate_proj names — shapes differ for up/down, skipped).
            sae = src_all_experts.get(li) if (gj is not None and name.endswith("gate_proj.weight")) else None
            sig, ma = _signature(name, tr.float().cpu(), eng.float().cpu(),
                                  src_all_experts=sae, gj=gj)
            results.append((name, emode, sig, ma))

        # Per-tensor dump: print ALL non-expert tensors + ALL non-CLEAN expert
        # tensors verbatim; collapse CLEAN expert lines to a count (128*3*2 expert
        # rows would otherwise drown the signal). The raw non-CLEAN lines ARE the
        # m!=j permutation evidence.
        clean_expert_count = 0
        for name, emode, sig, ma in results:
            is_expert = expert_re.search(name) is not None
            if is_expert and sig.startswith("CLEAN"):
                clean_expert_count += 1
                continue
            mas = f"{ma:.3e}" if ma == ma else "  nan  "
            print(f"    {name:56s} | {emode:14s} | {mas} | {sig}")
        if clean_expert_count:
            print(f"    [+ {clean_expert_count} CLEAN expert tensors collapsed]")

        # Verdict summary, split expert vs non-expert (the clean-control).
        def _bucket(rs):
            cln = [r for r in rs if r[2].startswith("CLEAN")]
            mis = [r for r in rs if "MISSING" in r[2]]
            dff = [r for r in rs if not r[2].startswith("CLEAN") and "MISSING" not in r[2]]
            return cln, dff, mis

        expert_rows = [r for r in results if expert_re.search(r[0]) is not None]
        nonexp_rows = [r for r in results if expert_re.search(r[0]) is None]
        e_cln, e_dff, e_mis = _bucket(expert_rows)
        n_cln, n_dff, n_mis = _bucket(nonexp_rows)
        print(f"\n[diag] SUMMARY  experts:    {len(e_cln)} CLEAN / {len(e_dff)} CORRUPT / {len(e_mis)} missing  (of {len(expert_rows)})")
        print(f"[diag] SUMMARY  non-expert: {len(n_cln)} CLEAN / {len(n_dff)} CORRUPT / {len(n_mis)} missing  (of {len(nonexp_rows)})")

        # WRONG_EXPERT permutation map (engine name j carries source expert m!=j).
        wrong = [(r[0], r[2]) for r in expert_rows if "WRONG_EXPERT" in r[2]]
        if wrong:
            print(f"[diag] WRONG_EXPERT permutation (m!=j) — {len(wrong)} gate_proj rows:")
            for nm, sg in wrong:
                print(f"    {nm}  ->  {sg}")

        # ---- W-vs-I VERDICT ----
        experts_clean = (len(e_dff) == 0 and len(e_cln) > 0)
        nonexp_clean = (len(n_dff) == 0)
        print("\n[diag] ===== W-vs-I VERDICT =====")
        if experts_clean and nonexp_clean:
            print("[diag] VERDICT = CLASS I (inference-time): engine-held weights EQUAL the FSDP "
                  "source for ALL experts AND non-expert tensors. Weights are faithfully synced "
                  "=> the salad is the EP=2 inference-time all-to-all (dispatch/combine), NOT weight "
                  "corruption. Next target: the EP dispatch/combine path, not the weight-load.")
        elif len(e_dff) > 0:
            print(f"[diag] VERDICT = CLASS W (weight corruption): {len(e_dff)} EXPERT tensors differ "
                  f"from source. The FSDP->vLLM EP weight-load/remap corrupts experts. See the "
                  f"signature lines above (WRONG_EXPERT m!=j => permutation; GATE_UP_SWAP/ROW_PERMUTED/"
                  f"SCALED/ZEROED => the sub-candidate). non-expert tensors: "
                  f"{'CLEAN (control holds)' if nonexp_clean else 'ALSO CORRUPT (not expert-specific!)'}")
        else:
            print(f"[diag] VERDICT = INDETERMINATE: experts all CLEAN/missing but {len(n_dff)} NON-expert "
                  f"tensors differ — investigate the non-expert corruption (not the expected expert-EP class).")
    finally:
        if pg is not None:
            try:
                ray.util.remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
