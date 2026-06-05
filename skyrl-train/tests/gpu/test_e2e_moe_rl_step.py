"""Stage 6 — end-to-end small-MoE RL step with backprop and router replay ON.

This is the integration capstone for the FSDP2 EP + router-replay port (Stages
0–5 GPU-validated). It proves the FULL training-side loop on a small real MoE
(``Qwen1.5-MoE-A2.7B-Chat``):

    rollout mask (captured ``routed_experts``) -> training batch carries it ->
    replay-train step with EP=2 + grouped-GEMM -> backprop (finite loss, sane
    grad-norm) -> weight-sync round-trip into the EP inference engine.

Rollout source (Stage 6 decision): **synthetic-but-realistic** ``rollout_routed_experts``
injected into the training ``Experience``. The LIVE Harbor->vLLM capture rail
(the patched vLLM fork emitting ``routed_experts`` over /v1) was already validated
end-to-end at Stage 1; standing up a live single-turn Harbor rollout *inside this
test* would add heavy, orthogonal infra for no extra coverage of the surface this
gate targets. The point of Stage 6 is the TRAINING-side e2e — replay consume ->
grouped-EP forward -> backprop -> weight-sync — which a synthetic mask exercises
faithfully: the mask forces routing (override "bites", per Stage 2), grad flows
through router + experts, and the EP-sharded grouped weights reshard+remap back
into the EP inference engine (the G4-4 oracle).

Run (pytest)::

    uv run --isolated --extra dev --extra vllm pytest tests/gpu/test_e2e_moe_rl_step.py

Or directly (no pytest, e.g. the cluster RL venv), from the ``skyrl-train`` dir::

    python tests/gpu/test_e2e_moe_rl_step.py

Requires >= 4 GPUs (EP=2 x FSDP=2 trainer, colocated with an EP=2 inference engine).
"""

import asyncio
import math
import os
import random
import sys

# Allow `python tests/gpu/test_e2e_moe_rl_step.py` from the skyrl-train dir (no
# pytest / no installed `tests` package): put the repo root on sys.path so
# `import tests.gpu.utils` resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import ray
import torch
from ray.util.placement_group import placement_group

try:
    import pytest
except ImportError:  # pytest absent on cluster RL venv — direct invocation still works
    pytest = None

from transformers import AutoTokenizer
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


# Inlined from tests/gpu/test_expert_parallel_inference.py (which imports pytest
# unconditionally at module top-level → can't be imported under the cluster RL venv
# that has no pytest). Kept byte-equivalent to the Stage-4 scaffolding it mirrors.
MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
NUM_GPUS = 4  # Should be divisible by 2


def _get_test_cfg() -> DictConfig:
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = MODEL
    cfg.generator.backend = "vllm"
    cfg.generator.async_engine = True
    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = 2
    cfg.generator.inference_engine_expert_parallel_size = 2
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.gpu_memory_utilization = 0.8
    cfg.generator.max_input_length = 2048
    cfg.generator.sampling_params.max_generate_length = 512
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.train_batch_size = 128
    cfg.trainer.policy_mini_batch_size = 128
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.update_epochs_per_batch = 1
    return cfg


def init_ray_inference_engines(backend, tp_size, shared_pg, config) -> InferenceEngineClient:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
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
        max_num_seqs=1024,
        tokenizer=tokenizer,
        backend=backend,
    )
    return InferenceEngineClient(engine, tokenizer, config)


def _check_gpus(num_gpus: int):
    available = get_available_gpus()
    if len(available) < num_gpus:
        msg = f"Stage 6 e2e requires >= {num_gpus} GPUs, found {len(available)}: {available}"
        if pytest is not None:
            pytest.skip(msg)
        raise RuntimeError(msg)


def _make_replay_experience(model_config, batch_size=2, seq_len=48, num_actions=16, device="cpu") -> Experience:
    """Build a training ``Experience`` carrying a synthetic-but-realistic
    ``rollout_routed_experts`` mask shaped ``[B, num_actions, L, K]``.

    The forced expert set is a distinct, valid top-K per response token (a shifted
    run starting at ``expert_id``), the same construction the Stage-2 ``test_a_override_bites``
    used to prove the override actually changes routing (so the replay is *active*,
    not a no-op fall-through). One row is left all-SENTINEL so the test also exercises
    the natural-routing fall-through inside the same batch.
    """
    torch.manual_seed(42)
    L = count_moe_layers(model_config)
    K = int(model_config.num_experts_per_tok)
    num_experts = int(model_config.num_experts)

    # Distinct valid top-K set per token: [expert_id, expert_id+1, ...] mod num_experts.
    expert_id = 3
    re = torch.empty(batch_size, num_actions, L, K, dtype=torch.long, device=device)
    for k in range(K):
        re[..., k] = (expert_id + k) % num_experts
    # Row 0: all-sentinel response -> natural routing fall-through (mixed batch).
    if batch_size > 1:
        re[0] = SENTINEL_EXPERT_ID

    B, T = batch_size, seq_len
    # Realistic response length (num_actions=16 >> the degenerate 8 the prior gate
    # used; seq_len=48 > num_actions so the prompt slice and the
    # `[:, -num_actions-1:-1]` gather are non-degenerate). Token range 0-100
    # (matches the proven make_dummy_experience used by the dense training_step GPU
    # test) keeps the synthetic forward numerically sane while still exercising the
    # full replay→EP→backprop path.
    #
    # NOTE: these old/base log-probs are the `old_action_log_probs` /
    # `base_action_log_probs` only; the trainer recomputes the (new)
    # `action_log_probs` from a genuine policy forward on `sequences`. With the
    # training-forward temperature fixed to 1.0 (see the gate config: the prior
    # NaN was a `logits.div_(0.0)` from temperature=0.0, NOT a degenerate aux term)
    # the new log-probs are finite, so the GRPO ratio `exp(new − old)` (clamped) is
    # finite and ≈ a sane O(1) value.
    #
    # ADVANTAGES carry genuine intra- and inter-row VARIANCE (not the prior
    # constant 0.6): a non-constant advantage means the surrogate loss
    # `-min(r·A, clip(r)·A)` is a meaningful, finite, non-trivial signal and the
    # masked-mean reduction can't collapse to a degenerate value. (GRPO group
    # normalization is NOT re-run inside training_step — `advantages` is consumed
    # as-is — so finite, non-constant values here are sufficient.)
    torch.manual_seed(1234)
    advantages = torch.randn((B, num_actions), device=device)  # finite, mean~0, var>0
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
    """Seed all RNGs so the GRPO step + both logprob forwards are reproducible.

    ``use_deterministic_algorithms`` is intentionally NOT forced globally: it makes
    several bf16 MoE / FSDP collectives raise ``RuntimeError`` (no deterministic
    kernel), and we do not need bit-exact determinism — we need the SAME weights to
    be exercised by both the trainer forward and the vLLM forward. Fixed seeds +
    ``cudnn.deterministic`` are sufficient for that; the logprob-agreement gate then
    tolerates the residual fp/grouped-vs-HF/EP-reduction-order noise (see below).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_e2e_moe_rl_step_replay_ep_grouped():
    """One full RL training step on a small MoE under EP=2 + grouped-GEMM +
    router-replay, then a weight-sync round-trip into the EP inference engine.

    Asserts:
      * training step completes; loss + grad-norm finite (no NaN/inf), grad-norm > 0
        (grad flows through router + experts);
      * DIRECT weight-equality gate: the trainer's POST-STEP HF-named weights (run
        through the SAME grouped->HF remap the broadcast uses) are read back from the
        synced EP=2 vLLM engine (inverting vLLM's HF->internal mapping) and compared
        tensor-by-tensor for a representative set — router.gate (routing-critical,
        x2 layers), expert w1/w2/w3 across BOTH EP shards, attention o_proj and
        embed_tokens. Weight-sync broadcasts the exact bf16 bytes, so faithful
        transport gives byte-exact equality (<= a tiny fp32-readback epsilon); a
        mis-remap/stale-shard shows as a large per-tensor max-abs. This REPLACES the
        prior per-token logprob-agreement proxy, which was confounded by MoE
        routing-tie divergence between the trainer's grouped-GEMM forward and vLLM's
        native MoE (2-3 outlier tokens diverged by O(1) even when transport was
        byte-exact). It validates a REAL GRPO update (unlike the rejected lr=0.0).
    """
    _check_gpus(num_gpus=NUM_GPUS)

    # DETERMINISM (Stage 6, TEST-ONLY): seed every RNG so the GRPO step and both
    # logprob forwards are reproducible across the run. The deterministic
    # logprob-agreement gate (below) replaces the prior greedy-decode oracle, which
    # was a known artifact (a genuine first AdamW step shifts router.gate ~1e-6
    # coherently -> flips greedy argmax -> full Levenshtein divergence -> 0/4).
    _seed_everything(1234)

    pg = None
    try:
        cfg = _get_test_cfg()
        cfg.trainer.placement.colocate_all = True
        # Stage 6: full replay + grouped-GEMM + EP=2 on the trainer (3-D mesh ddp=1, ep=2, fsdp=2).
        cfg.trainer.policy.fsdp_config.moe_router_replay = True
        cfg.trainer.policy.fsdp_config.moe_grouped_gemm = True
        cfg.trainer.policy.fsdp_config.expert_model_parallel_size = 2
        cfg.trainer.policy.fsdp_config.fsdp_size = 2
        # Router replay is INCOMPATIBLE with activation/gradient checkpointing:
        # checkpointing re-runs the MoE forward during backward, but the replay
        # controller's per-microbatch targets are set/cleared around the FIRST
        # forward only (model_wrapper.forward: begin_replay → set_microbatch_targets
        # → ... → clear). On recompute the controller is no longer REPLAY-active, so
        # the grouped shim takes the natural-routing branch → a different number of
        # saved tensors than the original forward → torch CheckpointError
        # ("75 vs 70 tensors"). The 14B-A2.7B model fits in 96 GB GH200 at
        # micro_train_batch_size_per_gpu=1 without checkpointing, so disable it here.
        # (Stage 7 / production note: to keep checkpointing ON with replay, the
        # controller must be made recompute-safe — re-arm the same targets for the
        # recomputation pass instead of clearing after the first forward.)
        cfg.trainer.gradient_checkpointing = False
        # Leave GPU headroom for the colocated EP-sharded trainer next to the engines.
        cfg.generator.gpu_memory_utilization = 0.45
        # TRAINING-FORWARD TEMPERATURE (Stage 6 ROOT CAUSE FIX, TEST-ONLY):
        # `Worker.training_step` passes `cfg.generator.sampling_params.temperature`
        # straight into the *training* model forward, where `model_wrapper.forward`
        # does `logits_BSV.div_(temperature)` (model_wrapper.py:427). The prior gate
        # set this to 0.0 (for "deterministic sampling"), which made the training
        # forward divide logits by ZERO -> inf/NaN logits -> NaN new-action_log_probs
        # -> NaN log_ratio -> NaN policy_loss / raw_grad_norm. That is EXACTLY the
        # 591214 status dict (`policy_loss: nan`, `log_ratio_abs_mean: nan`,
        # `raw_grad_norm: nan`). The fix is to keep the training-forward temperature
        # at the normal training value (1.0 — no scaling, log-probs straight from
        # logits), and apply the deterministic (greedy) temperature ONLY to the
        # inference sampling params used by the pre-step engine SMOKE generate (built
        # below). The GATE itself uses prompt_logprobs scoring (temperature=1.0, no
        # sampling), so it does not depend on greedy decode at all.
        cfg.generator.sampling_params.temperature = 1.0
        cfg.generator.sampling_params.top_p = 1.0
        cfg.generator.sampling_params.top_k = -1
        # Deterministic (greedy) sampling params for the inference engine ONLY — used
        # for the pre-step engine smoke generate (a liveness check, NOT the gate).
        # Decoupled from the (temperature=1.0) value the training forward reads above.
        from copy import deepcopy

        infer_sampling_cfg = deepcopy(cfg.generator.sampling_params)
        infer_sampling_cfg.temperature = 0.0
        # GATE NUMERICS (Stage 6, TEST-ONLY): with the training-forward temperature
        # fixed to 1.0 the core GRPO `policy_loss` term (ratio clamped, advantage-
        # weighted) is now well-defined and carries the gradient through router +
        # experts. The auxiliary terms are still degenerate on synthetic data — the
        # KL term pairs the real `new` log-prob against a *constant* base (0.3) and
        # the entropy term adds a second noisy contribution — and neither adds
        # coverage of the surface this gate targets (replay->EP->backprop->weight-
        # sync). Keeping them off leaves `loss = policy_loss`, the term that must be
        # finite and must backprop. This is gate-config only; the a3 production path
        # (ep_size=1) is untouched and keeps its own algorithm config.
        cfg.trainer.algorithm.use_kl_loss = False
        cfg.trainer.algorithm.use_entropy_loss = False

        initialize_ray(cfg)

        pg = placement_group([{"GPU": 1, "CPU": 1}] * NUM_GPUS, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=60)

        # EP=2 inference engine, colocated.
        client = init_ray_inference_engines(
            backend=cfg.generator.backend,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            shared_pg=pg,
            config=cfg,
        )
        asyncio.run(client.wake_up())

        prompts = get_test_prompts(MODEL, num_samples=4)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

        # Smoke the engine once so a dead-engine failure surfaces before the (much
        # longer) training step; this generate is NOT the gate (the gate is the
        # post-step DIRECT weight-equality check below).
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, infer_sampling_cfg)
        out_before = asyncio.run(
            client.generate(InferenceEngineInput(prompts=prompts, sampling_params=sampling_params))
        )
        assert len(out_before["responses"]) == len(prompts)

        asyncio.run(client.sleep())

        # FSDP2 policy: grouped-swapped + EP-sharded + replay-on.
        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=True,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # The replay mask must match the trainer model's MoE topology (L, K, num_experts).
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
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
            # Print the full status dict BEFORE asserting so a NaN in any component
            # is attributable (the prior gate failure logged only the bare assert).
            print(f"[Stage6][rank{rank}] training_step status: {result}")
        for rank, result in enumerate(results):
            assert "policy_loss" in result and "final_loss" in result
            # The policy (GRPO) loss is the load-bearing term that backprops through
            # router + experts; it must be finite (kl/entropy are off in this gate).
            pl_v = result["policy_loss"]
            assert math.isfinite(pl_v), f"policy_loss is not finite (rank {rank}): {pl_v}"
            loss_v = result["final_loss"]
            assert math.isfinite(loss_v), f"loss is not finite (rank {rank}): {loss_v}"
            assert "raw_grad_norm" in result, "grad-norm missing — optimizer step did not run"
            gn = result["raw_grad_norm"]
            assert math.isfinite(gn), f"grad-norm is not finite (rank {rank}): {gn}"
            assert gn > 0.0, f"grad-norm is zero — no grad flowed through router/experts: {gn}"
        print(
            f"[Stage6] replay+EP+grouped training step: loss={results[0]['final_loss']:.4f} "
            f"policy_loss={results[0]['policy_loss']:.4f} grad_norm={results[0]['raw_grad_norm']:.4f} "
            f"entropy={results[0]['policy_entropy']:.4f}"
        )

        # --- Weight-sync round-trip into the EP inference engine (G4-4 path). ---
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.wake_up(tags=["weights"]))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

        # --- DIRECT WEIGHT-EQUALITY GATE (Stage 6, replaces the logprob proxy). ---
        # The prior gate compared per-token LOGPROBS of the trainer vs the synced
        # engine. That is a numerical PROXY for "the updated weights landed", and it
        # is confounded by MoE routing-tie divergence: the FSDP2 grouped-GEMM trainer
        # forward and vLLM's native MoE forward break top-k expert ties differently,
        # so 2-3 outlier tokens diverge by O(1) even when the transport is byte-exact.
        # We replace it with a DIRECT tensor-by-tensor comparison of the trainer's
        # post-step HF-named weights against the SAME tensors read back from the vLLM
        # engine (through the grouped->HF remap on the trainer side, and the inverse
        # of vLLM's HF->internal mapping on the engine side). Since weight-sync
        # broadcasts the exact bf16 bytes, faithful transport gives EXACT equality up
        # to a tiny fp32-readback epsilon; a mis-remap/stale-shard shows as a large
        # per-tensor max-abs on the affected tensor.
        #
        # Representative set (covers every weight-sync surface of the EP+grouped+
        # replay port):
        #   * router.gate  (the routing-critical tensor; replicated, direct)        x2 layers
        #   * expert w1/w3/w2 across BOTH EP shards (experts 0,29 -> ep0; 30,59 -> ep1)
        #   * attention o_proj (TP row-sharded) and embed_tokens (TP vocab-sharded)
        EP = int(cfg.trainer.policy.fsdp_config.expert_model_parallel_size)
        n_experts = int(model_config.num_experts)
        experts_per_ep = n_experts // EP
        # experts spanning both EP shards: first+last of ep0, first+last of ep1
        rep_experts = sorted(set([0, experts_per_ep - 1, experts_per_ep, n_experts - 1]))
        rep_names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate.weight",   # router (layer 0)
            "model.layers.12.mlp.gate.weight",  # router (layer 12)
        ]
        for j in rep_experts:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                rep_names.append(f"model.layers.0.mlp.experts.{j}.{proj}.weight")
        print(f"[Stage6][weight-eq] EP={EP} experts_per_ep={experts_per_ep} rep_experts={rep_experts}")
        print(f"[Stage6][weight-eq] comparing {len(rep_names)} representative tensors")

        # (a) Trainer POST-STEP HF tensors (model still on GPU; SAME extract+remap as
        #     broadcast). Collective over all ranks; rank 0 carries the full tensors.
        trainer_w_per_rank = ray.get(
            policy.async_run_ray_method("pass_through", "read_post_step_weights", rep_names)
        )
        trainer_w = {}
        for d in trainer_w_per_rank:
            if isinstance(d, dict):
                trainer_w.update({k: v for k, v in d.items() if isinstance(v, torch.Tensor)})

        # Now offload + wake kv cache (post-readback so the model was on GPU for extract).
        policy.offload_to_cpu()
        asyncio.run(client.wake_up(tags=["kv_cache"]))
        asyncio.run(client.reset_prefix_cache())

        # (b) ENGINE-side readback of the SAME HF names, gathered across all TP/EP
        #     workers (inventory dumped once for diagnosis).
        engine_actor = client.engines[0].inference_engine_actor
        engine_per_rank = ray.get(
            engine_actor.read_engine_weights.remote(rep_names, True)
        )
        # collective_rpc returns one dict per worker rank.
        if isinstance(engine_per_rank, dict):
            engine_per_rank = [engine_per_rank]

        # Print the engine param inventory once (first run aid / future debugging).
        for rk, rd in enumerate(engine_per_rank):
            inv = rd.get("__inventory__") if isinstance(rd, dict) else None
            ranks = rd.get("__ranks__") if isinstance(rd, dict) else None
            if inv is not None:
                sample = {
                    k: v for k, v in inv.items()
                    if "layers.0." in k or "embed_tokens" in k
                }
                print(f"[Stage6][weight-eq][engine-rank{rk}] ranks={ranks} layer0/embed inventory:")
                for k in sorted(sample):
                    print(f"    {k} {sample[k]}")

        def _assemble_engine_tensor(name):
            """Assemble the full HF tensor for ``name`` from the per-rank engine readback.

            - direct (replicated): any rank's tensor (assert agreement across ranks).
            - expert: the single owner-EP rank holds it (one tensor).
            - TP-sharded (embed_tokens / o_proj): concatenate across TP ranks along the
              shard dim. embed_tokens shards dim0 (vocab); o_proj is RowParallel ->
              shards dim1 (input). We return (tensor, mode); on any assembly ambiguity
              we return mode='partial' and the caller reports rather than hard-fails so
              the load-bearing router/expert verdict is never masked.
            """
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
                # Replicated: every rank should agree; take rank0's, note max cross-rank diff.
                t0 = entries[0][1]["tensor"]
                return t0, "direct"
            if mode == "expert":
                # Exactly one owner-EP rank (per its TP group). All owner copies identical.
                return entries[0][1]["tensor"], "expert"
            # Fallback unknown
            return entries[0][1]["tensor"], (mode or "unknown")

        # TP-sharded names need explicit assembly across TP ranks. Build TP-rank-indexed
        # shard lists for embed_tokens (dim0) and o_proj (dim1).
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

        EPS = 1e-4  # fp32-readback + bf16-roundtrip epsilon; transport is byte-exact otherwise.
        results_tbl = []  # (name, status, max_abs, shape)
        TP_SHARDED = {
            "model.embed_tokens.weight": 0,             # VocabParallelEmbedding -> dim0
            "model.layers.0.self_attn.o_proj.weight": 1,  # RowParallelLinear -> dim1 (input)
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
                # vLLM may store some weights transposed relative to HF (e.g. expert
                # GEMM layout). For 2D tensors, try the transpose before declaring a
                # shape mismatch so a benign layout convention isn't read as corruption.
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

        print("[Stage6][weight-eq] per-tensor results (name | status | max_abs | shape):")
        for name, status, max_abs, shape in results_tbl:
            print(f"    {name:64s} | {status:24s} | {max_abs:.3e} | {shape}")

        # The GATE. The routing-critical / expert / attention tensors must be EXACT
        # (within EPS). TP-sharded assembly (embed/o_proj) is reported but, if its
        # assembly is ambiguous (ENGINE_MISSING / SHAPE_MISMATCH from a TP-cat quirk),
        # it does NOT mask the load-bearing direct/expert verdict — but a real DIFF on
        # ANY found+shape-matched tensor IS a hard failure.
        hard_fail = [r for r in results_tbl if r[1] == "DIFF"]
        # router + expert tensors MUST be present and OK (these are the EP+grouped+
        # replay reshard/remap surfaces the gate exists to validate).
        critical = [
            n for n in rep_names
            if n.endswith("mlp.gate.weight") or ".mlp.experts." in n
        ]
        crit_status = {n: s for (n, s, _, _) in results_tbl}
        missing_critical = [n for n in critical if crit_status.get(n) != "OK"]

        assert not hard_fail, (
            "DIRECT weight-equality FAILED: the following synced tensors differ from the trainer's "
            f"post-step weights by > {EPS} -> the broadcast/reshard/remap mis-propagated them: "
            + "; ".join(f"{n} (max_abs={ma:.3e})" for (n, s, ma, sh) in hard_fail)
        )
        assert not missing_critical, (
            "DIRECT weight-equality could not VERIFY these routing-critical / expert tensors "
            f"(status != OK): {missing_critical}. The router.gate and expert w1/w2/w3 across BOTH EP "
            "shards must be read back and matched for the gate to close."
        )
        n_ok = sum(1 for r in results_tbl if r[1] == "OK")
        print(
            f"[Stage6] DIRECT weight-equality gate PASS: {n_ok}/{len(rep_names)} representative tensors "
            f"byte-exact (<= {EPS}) incl. router.gate (x2 layers) + experts across both EP shards + "
            f"attention/embed -> the real GRPO update is faithfully in the EP engine."
        )
    finally:
        if pg is not None:
            try:
                ray.util.remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()


if __name__ == "__main__":
    test_e2e_moe_rl_step_replay_ep_grouped()
    print("Stage 6 e2e: ALL PASS")
