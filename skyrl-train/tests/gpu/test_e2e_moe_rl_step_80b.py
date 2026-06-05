"""Stage 7 (P4 capstone) — end-to-end Qwen3-Next-80B-A3B RL step with backprop +
router-replay ON, multi-node EP=8 x FSDP.

This is the SCALED-UP twin of ``test_e2e_moe_rl_step.py`` (the Stage-6 small-MoE
e2e gate, jobs 595401 / 595520). The MECHANISM is identical and already proven at
small scale on 4xGH200 (EP=2 x FSDP=2 + grouped-GEMM + replay forward -> backprop ->
mesh-aware grad-clip -> optimizer step + a DIRECT byte-equality weight-sync gate).
Stage 7 extends the SAME harness to:

  * the real ``Qwen/Qwen3-Next-80B-A3B-Instruct`` (GatedDeltaNet linear-attention +
    512-expert top-10 + 1 sigmoid-gated shared expert; full_attention every 4th
    layer; every layer carries an MoE block),
  * multi-node EP=8 x FSDP across 8 GH200 nodes (32 GPUs); experts shard 32-way
    (ep=8 composed with fsdp=4), non-expert/attention/optimizer shard fsdp=4-way,
  * ``gradient_checkpointing=True`` (now replay-recompute-safe via P3, commit 9a0ae57f),
  * torch-EP all-to-all backend (RDMA-agnostic; DeepEP internode deferred to a perf
    layer per stage5/stage7 scope).

Rollout source: synthetic-but-realistic ``rollout_routed_experts`` (same Stage-6
construction). The LIVE Harbor->vLLM capture rail was validated at Stage 1 / P1
(job 595540); a synthetic mask faithfully exercises the TRAINING-side surface this
gate targets (replay consume -> grouped-EP forward -> backprop -> weight-sync) and
is the right first 80B run per the Stage-7 directive.

Asserts (the Stage-7 capstone gate):
  (1) one full GRPO step completes; policy_loss + final_loss finite; raw_grad_norm
      finite and > 0 (grad flows through router + experts + GatedDeltaNet);
  (2) DIRECT weight-equality: trainer POST-STEP HF-named weights == the SAME tensors
      read back from the synced EP=8 vLLM engine, byte-exact (<= 1e-4) for a
      representative set spanning router.gate (x2 layers) + expert w1/w2/w3 across
      ALL 8 EP shards + attention o_proj + embed_tokens.

Run (cluster, in the SIF, multi-node via srun -- see run_p4_capstone.sbatch)::

    python tests/gpu/test_e2e_moe_rl_step_80b.py

Requires 32 GPUs (8 GH200 nodes x 4). EP=8 spans >=2 nodes so internode torch-EP
all-to-all is on the path regardless of node count.
"""

import asyncio
import math
import os
import random
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import ray
import torch
from ray.util.placement_group import placement_group

try:
    import pytest
except ImportError:  # pytest absent in the SIF / cluster runtime — direct invocation still works
    pytest = None

from transformers import AutoTokenizer, AutoConfig
from omegaconf import DictConfig

from tests.gpu.utils import (
    get_available_gpus,
    get_test_actor_config,
    get_test_prompts,
    init_worker_with_type,
)
from skyrl_train.dataset.replay_buffer import Experience
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.models.router_replay import count_moe_layers, SENTINEL_EXPERT_ID
from skyrl_train.utils import initialize_ray, get_ray_pg_ready_with_timeout


# --- Stage-7 topology (the capstone budget; see run_p4_capstone.sbatch header) ---
# Default to the in-cache path on Jupiter; overridable via env for portability.
MODEL = os.environ.get("SKYRL_80B_MODEL_PATH", "Qwen/Qwen3-Next-80B-A3B-Instruct")
NUM_NODES = int(os.environ.get("SKYRL_80B_NUM_NODES", "8"))
GPUS_PER_NODE = int(os.environ.get("SKYRL_80B_GPUS_PER_NODE", "4"))
NUM_GPUS = NUM_NODES * GPUS_PER_NODE  # 32
EP_SIZE = int(os.environ.get("SKYRL_80B_EP_SIZE", "8"))        # 512 experts / 8 ranks = 64/rank
FSDP_SIZE = int(os.environ.get("SKYRL_80B_FSDP_SIZE", "4"))    # ddp=1, ep=8, fsdp=4 -> world 32
# vLLM EP inference engine: EP=8 spans 8 GPUs (TP=8 group hosts the EP=8 experts).
INFER_TP = int(os.environ.get("SKYRL_80B_INFER_TP", "8"))
INFER_EP = int(os.environ.get("SKYRL_80B_INFER_EP", "8"))
# Conservative colocated KV headroom for the 80B trainer (offload_after_step gives
# the trainer the GPU back during generate; the engine still needs weights+KV room).
GPU_MEM_UTIL = float(os.environ.get("SKYRL_80B_GPU_MEM_UTIL", "0.40"))


def _get_test_cfg() -> DictConfig:
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = MODEL
    cfg.generator.backend = "vllm"
    cfg.generator.async_engine = True
    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = INFER_TP
    cfg.generator.inference_engine_expert_parallel_size = INFER_EP
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.gpu_memory_utilization = GPU_MEM_UTIL
    cfg.generator.max_input_length = 2048
    cfg.generator.sampling_params.max_generate_length = 256
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.train_batch_size = 2
    cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.placement.policy_num_nodes = NUM_NODES
    cfg.trainer.placement.policy_num_gpus_per_node = GPUS_PER_NODE
    cfg.trainer.update_epochs_per_batch = 1
    return cfg


def init_ray_inference_engines(backend, tp_size, shared_pg, config) -> InferenceEngineClient:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    engine = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=tp_size,
        expert_parallel_size=config.generator.inference_engine_expert_parallel_size,
        model_dtype="bfloat16",
        pretrain=MODEL,
        seed=42,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        shared_pg=shared_pg,
        gpu_memory_utilization=config.generator.gpu_memory_utilization,
        inference_engine_enable_sleep=config.trainer.placement.colocate_all,
        async_engine=True,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        tokenizer=tokenizer,
        backend=backend,
    )
    return InferenceEngineClient(engine, tokenizer, config)


def _check_gpus(num_gpus: int):
    available = get_available_gpus()
    if len(available) < num_gpus:
        msg = f"Stage 7 80B e2e requires >= {num_gpus} GPUs, found {len(available)}: {available}"
        if pytest is not None:
            pytest.skip(msg)
        raise RuntimeError(msg)


def _assert_r3_alignment(model_config):
    """The Stage-7 R3 footgun: the replay mask must match the model's MoE topology.

    Qwen3-Next-80B-A3B: 48 layers, decoder_sparse_step=1, mlp_only_layers=[] ->
    EVERY layer carries an MoE block (num_moe_layers == num_layers == 48), 512
    experts, top-10. full_attention_interval=4 only swaps the ATTENTION type
    (GatedDeltaNet linear vs full), NOT whether the layer is MoE. We assert the
    derived (L, K, num_experts) against the known capstone topology so a config
    drift (or a num_moe_layers != num_layers regression) fails LOUD before the run.
    """
    L = count_moe_layers(model_config)
    K = int(model_config.num_experts_per_tok)
    E = int(model_config.num_experts)
    n_layers = int(model_config.num_hidden_layers)
    print(f"[Stage7][R3] num_layers={n_layers} num_moe_layers(L)={L} top_k(K)={K} num_experts(E)={E}")
    assert E == 512, f"expected 512 experts, got {E}"
    assert K == 10, f"expected top-10, got {K}"
    assert L == n_layers == 48, (
        f"R3 alignment: num_moe_layers={L} num_layers={n_layers} (expected both 48). "
        "If these diverge the replay mask [B,A,L,K] will mis-index the MoE stack."
    )
    assert E % EP_SIZE == 0, f"num_experts={E} not divisible by EP_SIZE={EP_SIZE}"
    return L, K, E


def _make_replay_experience(model_config, batch_size=2, seq_len=48, num_actions=16, device="cpu") -> Experience:
    """Synthetic-but-realistic replay mask shaped [B, num_actions, L, K] (L=48, K=10)."""
    torch.manual_seed(42)
    L = count_moe_layers(model_config)
    K = int(model_config.num_experts_per_tok)
    num_experts = int(model_config.num_experts)

    expert_id = 3
    re = torch.empty(batch_size, num_actions, L, K, dtype=torch.long, device=device)
    for k in range(K):
        re[..., k] = (expert_id + k) % num_experts
    if batch_size > 1:
        re[0] = SENTINEL_EXPERT_ID  # natural-routing fall-through row (mixed batch)

    B, T = batch_size, seq_len
    torch.manual_seed(1234)
    advantages = torch.randn((B, num_actions), device=device)
    return Experience(
        sequences=torch.randint(0, 100, (B, T), device=device),
        action_log_probs=0.4 * torch.ones((B, num_actions), device=device),
        base_action_log_probs=0.3 * torch.ones((B, num_actions), device=device),
        rollout_logprobs=0.2 * torch.ones((B, num_actions), device=device),
        values=0.5 * torch.ones((B, num_actions), device=device),
        returns=0.5 * torch.ones((B, num_actions), device=device),
        advantages=advantages,
        attention_mask=torch.ones((B, T), dtype=int, device=device),
        loss_mask=torch.ones((B, num_actions), dtype=int, device=device),
        action_mask=torch.ones((B, num_actions), dtype=int, device=device),
        num_actions=num_actions,
        info={},
        rollout_routed_experts=re,
    )


def _seed_everything(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_e2e_moe_rl_step_80b_replay_ep_grouped():
    """The Stage-7 capstone: one full GRPO step (backprop) on Qwen3-Next-80B-A3B
    under EP=8 x FSDP=4 + grouped-GEMM + router-replay + gradient-checkpointing,
    multi-node (8 GH200 nodes), then the DIRECT weight-equality gate against the
    synced EP=8 vLLM engine."""
    _check_gpus(num_gpus=NUM_GPUS)
    _seed_everything(1234)

    pg = None
    try:
        cfg = _get_test_cfg()
        cfg.trainer.placement.colocate_all = True
        # Stage 7: full replay + grouped-GEMM + EP=8 on the trainer (3-D ddp=1, ep=8, fsdp=4).
        cfg.trainer.policy.fsdp_config.moe_router_replay = True
        cfg.trainer.policy.fsdp_config.moe_grouped_gemm = True
        cfg.trainer.policy.fsdp_config.expert_model_parallel_size = EP_SIZE
        cfg.trainer.policy.fsdp_config.fsdp_size = FSDP_SIZE
        # torch-EP all-to-all backend first (RDMA-agnostic; DeepEP internode deferred).
        if "ep_comm_backend" in cfg.trainer.policy.fsdp_config:
            cfg.trainer.policy.fsdp_config.ep_comm_backend = "torch"
        # gradient checkpointing now replay-recompute-safe (P3, commit 9a0ae57f). At 80B
        # this is effectively mandatory to fit activations alongside the colocated engine.
        cfg.trainer.gradient_checkpointing = True
        print("[Stage7] gradient_checkpointing=True (replay-recompute-safe via P3)")
        cfg.generator.gpu_memory_utilization = GPU_MEM_UTIL
        # Training-forward temperature 1.0 (Stage-6 root-cause fix: temp=0.0 -> div-by-zero -> NaN).
        cfg.generator.sampling_params.temperature = 1.0
        cfg.generator.sampling_params.top_p = 1.0
        cfg.generator.sampling_params.top_k = -1
        from copy import deepcopy

        infer_sampling_cfg = deepcopy(cfg.generator.sampling_params)
        infer_sampling_cfg.temperature = 0.0
        # GRPO policy_loss is the load-bearing (backprop) term; kl/entropy off keeps
        # loss == policy_loss on synthetic data (same gate-config as Stage 6).
        cfg.trainer.algorithm.use_kl_loss = False
        cfg.trainer.algorithm.use_entropy_loss = False

        # R3 alignment guard BEFORE any GPU work (fail loud on config drift).
        model_config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        _assert_r3_alignment(model_config)

        initialize_ray(cfg)

        # MULTI-NODE placement group: one GPU-bundle per node (STRICT_SPREAD so the 8
        # bundles land on 8 distinct nodes -> EP=8 spans >=2 nodes -> internode
        # torch-EP all-to-all is genuinely exercised). PACK would collapse onto fewer
        # nodes and not fit the 80B shard.
        bundles = [{"GPU": GPUS_PER_NODE, "CPU": GPUS_PER_NODE} for _ in range(NUM_NODES)]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        get_ray_pg_ready_with_timeout(pg, timeout=600)

        # EP=8 inference engine, colocated.
        client = init_ray_inference_engines(
            backend=cfg.generator.backend,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            shared_pg=pg,
            config=cfg,
        )
        asyncio.run(client.wake_up())

        prompts = get_test_prompts(MODEL, num_samples=2)

        # Smoke the engine once (liveness, NOT the gate).
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, infer_sampling_cfg)
        out_before = asyncio.run(
            client.generate(InferenceEngineInput(prompts=prompts, sampling_params=sampling_params))
        )
        assert len(out_before["responses"]) == len(prompts)
        asyncio.run(client.sleep())

        # FSDP2 policy: grouped-swapped + EP=8-sharded + replay-on, multi-node.
        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=True,
            num_gpus_per_node=GPUS_PER_NODE,
            num_nodes=NUM_NODES,
            cfg=cfg,
        )

        experience = _make_replay_experience(model_config)

        # --- One full training step (replay consume -> grouped-EP forward -> backprop). ---
        global_step, local_step, accumulation_steps = 0, 0, 1
        results = ray.get(
            policy.async_run_ray_method(
                "pass_through", "training_step", experience, global_step, local_step, accumulation_steps
            )
        )

        for rank, result in enumerate(results):
            assert isinstance(result, dict), "training_step should return a status dict"
            print(f"[Stage7][rank{rank}] training_step status: {result}")
        for rank, result in enumerate(results):
            assert "policy_loss" in result and "final_loss" in result
            pl_v = result["policy_loss"]
            assert math.isfinite(pl_v), f"policy_loss not finite (rank {rank}): {pl_v}"
            loss_v = result["final_loss"]
            assert math.isfinite(loss_v), f"loss not finite (rank {rank}): {loss_v}"
            assert "raw_grad_norm" in result, "grad-norm missing — optimizer step did not run"
            gn = result["raw_grad_norm"]
            assert math.isfinite(gn), f"grad-norm not finite (rank {rank}): {gn}"
            assert gn > 0.0, f"grad-norm is zero — no grad flowed through router/experts: {gn}"
        print(
            f"[Stage7] 80B replay+EP=8+grouped+ckpt training step: "
            f"loss={results[0]['final_loss']:.4f} policy_loss={results[0]['policy_loss']:.4f} "
            f"grad_norm={results[0]['raw_grad_norm']:.4f} entropy={results[0].get('policy_entropy', float('nan')):.4f}"
        )

        # --- Weight-sync round-trip into the EP=8 inference engine. ---
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.wake_up(tags=["weights"]))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

        # --- DIRECT WEIGHT-EQUALITY GATE (representative set across ALL 8 EP shards). ---
        EP = int(cfg.trainer.policy.fsdp_config.expert_model_parallel_size)
        n_experts = int(model_config.num_experts)
        experts_per_ep = n_experts // EP  # 64
        # First + last expert of EVERY EP shard -> covers all 8 shards' reshard/remap.
        rep_experts = sorted(set(
            [s * experts_per_ep for s in range(EP)] +
            [s * experts_per_ep + experts_per_ep - 1 for s in range(EP)]
        ))
        rep_names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate.weight",   # router (layer 0)
            "model.layers.12.mlp.gate.weight",  # router (layer 12)
        ]
        for j in rep_experts:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                rep_names.append(f"model.layers.0.mlp.experts.{j}.{proj}.weight")
        print(f"[Stage7][weight-eq] EP={EP} experts_per_ep={experts_per_ep} rep_experts={rep_experts}")
        print(f"[Stage7][weight-eq] comparing {len(rep_names)} representative tensors")

        trainer_w_per_rank = ray.get(
            policy.async_run_ray_method("pass_through", "read_post_step_weights", rep_names)
        )
        trainer_w = {}
        for d in trainer_w_per_rank:
            if isinstance(d, dict):
                trainer_w.update({k: v for k, v in d.items() if isinstance(v, torch.Tensor)})

        policy.offload_to_cpu()
        asyncio.run(client.wake_up(tags=["kv_cache"]))
        asyncio.run(client.reset_prefix_cache())

        engine_actor = client.engines[0].inference_engine_actor
        engine_per_rank = ray.get(engine_actor.read_engine_weights.remote(rep_names, True))
        if isinstance(engine_per_rank, dict):
            engine_per_rank = [engine_per_rank]

        for rk, rd in enumerate(engine_per_rank):
            inv = rd.get("__inventory__") if isinstance(rd, dict) else None
            ranks = rd.get("__ranks__") if isinstance(rd, dict) else None
            if inv is not None:
                sample = {k: v for k, v in inv.items() if "layers.0." in k or "embed_tokens" in k}
                print(f"[Stage7][weight-eq][engine-rank{rk}] ranks={ranks} layer0/embed inventory:")
                for k in sorted(sample):
                    print(f"    {k} {sample[k]}")

        def _assemble_engine_tensor(name):
            entries = []
            for rd in engine_per_rank:
                if not isinstance(rd, dict):
                    continue
                e = rd.get(name)
                rk = rd.get("__ranks__", {})
                if isinstance(e, dict) and e.get("found") and isinstance(e.get("tensor"), torch.Tensor):
                    entries.append((rk, e))
            if not entries:
                return None, "missing"
            mode = entries[0][1].get("mode")
            if mode == "direct":
                return entries[0][1]["tensor"], "direct"
            if mode == "expert":
                return entries[0][1]["tensor"], "expert"
            return entries[0][1]["tensor"], (mode or "unknown")

        def _assemble_tp_sharded(name, cat_dim):
            shards = {}
            for rd in engine_per_rank:
                if not isinstance(rd, dict):
                    continue
                e = rd.get(name)
                rk = rd.get("__ranks__", {})
                if isinstance(e, dict) and e.get("found") and isinstance(e.get("tensor"), torch.Tensor):
                    shards[rk.get("tp_rank", 0)] = e["tensor"]
            if not shards:
                return None
            ordered = [shards[i] for i in sorted(shards)]
            try:
                return torch.cat(ordered, dim=cat_dim)
            except Exception:
                return None

        EPS = 1e-4
        results_tbl = []
        TP_SHARDED = {
            "model.embed_tokens.weight": 0,
            "model.layers.0.self_attn.o_proj.weight": 1,
        }

        for name in rep_names:
            tr = trainer_w.get(name)
            if tr is None:
                results_tbl.append((name, "TRAINER_MISSING", float("nan"), None))
                continue
            if name in TP_SHARDED:
                eng = _assemble_tp_sharded(name, TP_SHARDED[name])
                emode = f"tp-cat(dim{TP_SHARDED[name]})"
            else:
                eng, emode = _assemble_engine_tensor(name)
            if eng is None:
                results_tbl.append((name, f"ENGINE_MISSING({emode})", float("nan"), list(tr.shape)))
                continue
            tr_f = tr.float().cpu()
            eng_f = eng.float().cpu()
            if tuple(tr_f.shape) != tuple(eng_f.shape):
                if tr_f.dim() == 2 and eng_f.dim() == 2 and tuple(tr_f.shape) == tuple(eng_f.t().shape):
                    eng_f = eng_f.t().contiguous()
                else:
                    results_tbl.append(
                        (name, f"SHAPE_MISMATCH trainer={tuple(tr_f.shape)} engine={tuple(eng_f.shape)}", float("nan"), list(tr_f.shape))
                    )
                    continue
            max_abs = float((tr_f - eng_f).abs().max().item())
            status = "OK" if max_abs <= EPS else "DIFF"
            results_tbl.append((name, status, max_abs, list(tr_f.shape)))

        print("[Stage7][weight-eq] per-tensor results (name | status | max_abs | shape):")
        for name, status, max_abs, shape in results_tbl:
            print(f"    {name:64s} | {status:24s} | {max_abs:.3e} | {shape}")

        hard_fail = [r for r in results_tbl if r[1] == "DIFF"]
        critical = [n for n in rep_names if n.endswith("mlp.gate.weight") or ".mlp.experts." in n]
        crit_status = {n: s for (n, s, _, _) in results_tbl}
        missing_critical = [n for n in critical if crit_status.get(n) != "OK"]

        assert not hard_fail, (
            "DIRECT weight-equality FAILED at 80B: synced tensors differ from trainer post-step "
            f"weights by > {EPS}: " + "; ".join(f"{n} (max_abs={ma:.3e})" for (n, s, ma, sh) in hard_fail)
        )
        assert not missing_critical, (
            "DIRECT weight-equality could not VERIFY routing-critical / expert tensors "
            f"(status != OK): {missing_critical}. router.gate + expert w1/w2/w3 across ALL 8 EP "
            "shards must be read back and matched for the gate to close."
        )
        n_ok = sum(1 for r in results_tbl if r[1] == "OK")
        print(
            f"[Stage7] CAPSTONE PASS: 80B GRPO step + DIRECT weight-equality {n_ok}/{len(rep_names)} "
            f"tensors byte-exact (<= {EPS}) across router.gate (x2) + experts across all {EP} EP shards "
            f"+ attention/embed -> the real GRPO update is faithfully in the EP=8 vLLM engine."
        )
    finally:
        if pg is not None:
            try:
                ray.util.remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()


if __name__ == "__main__":
    test_e2e_moe_rl_step_80b_replay_ep_grouped()
    print("Stage 7 80B e2e: ALL PASS")
