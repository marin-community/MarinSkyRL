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
import sys

# Allow `python tests/gpu/test_e2e_moe_rl_step.py` from the skyrl-train dir (no
# pytest / no installed `tests` package): put the repo root on sys.path so
# `import tests.gpu.utils` resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
    are_responses_similar,
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


def test_e2e_moe_rl_step_replay_ep_grouped():
    """One full RL training step on a small MoE under EP=2 + grouped-GEMM +
    router-replay, then a weight-sync round-trip into the EP inference engine.

    Asserts:
      * training step completes; loss + grad-norm finite (no NaN/inf), grad-norm > 0
        (grad flows through router + experts);
      * the weight-sync round-trips into the EP=2 inference engine and reproduces
        the pre-sync responses (the G4-4 ``are_responses_similar`` oracle).
    """
    _check_gpus(num_gpus=NUM_GPUS)

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
        # inference sampling params used by the weight-sync oracle (built below).
        cfg.generator.sampling_params.temperature = 1.0
        cfg.generator.sampling_params.top_p = 1.0
        cfg.generator.sampling_params.top_k = -1
        # Deterministic (greedy) sampling params for the inference engine ONLY — used
        # by the before/after weight-sync `are_responses_similar` oracle so the
        # round-trip comparison is stable. Decoupled from the (temperature=1.0)
        # value the training forward reads above.
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

        # --- Weight-sync round-trip into the EP inference engine (G4-4 oracle). ---
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.wake_up(tags=["weights"]))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        policy.offload_to_cpu()
        asyncio.run(client.wake_up(tags=["kv_cache"]))
        asyncio.run(client.reset_prefix_cache())

        out_after = asyncio.run(
            client.generate(InferenceEngineInput(prompts=prompts, sampling_params=sampling_params))
        )
        assert len(out_after["responses"]) == len(prompts)

        # After ONE GRPO step the weights have moved slightly; the round-trip must
        # still reshard+remap the EP-sharded grouped trainer weights faithfully into
        # the EP inference engine (responses stay close to pre-sync). Tolerance is
        # looser than the static G4-4 sync (0.02) because a real optimizer step ran.
        num_similar = sum(
            1
            for i in range(len(prompts))
            if are_responses_similar([out_before["responses"][i]], [out_after["responses"][i]], tolerance=0.15)
        )
        assert num_similar == len(prompts), (
            f"weight-sync round-trip corrupted weights: only {num_similar}/{len(prompts)} responses matched."
        )
        print(f"[Stage6] weight-sync round-trip into EP engine: {num_similar}/{len(prompts)} similar PASS")
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
