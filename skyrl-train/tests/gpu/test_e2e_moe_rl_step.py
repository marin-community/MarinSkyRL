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

Run::

    uv run --isolated --extra dev --extra vllm pytest tests/gpu/test_e2e_moe_rl_step.py

Requires >= 4 GPUs (EP=2 x FSDP=2 trainer, colocated with an EP=2 inference engine).
"""

import asyncio
import math

import pytest
import ray
import torch
from ray.util.placement_group import placement_group

from tests.gpu.utils import (
    are_responses_similar,
    get_available_gpus,
    get_test_prompts,
    init_worker_with_type,
)
from tests.gpu.test_expert_parallel_inference import (
    MODEL,
    NUM_GPUS,
    _get_test_cfg,
    init_ray_inference_engines,
)
from skyrl_train.dataset.replay_buffer import Experience
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.models.router_replay import count_moe_layers, SENTINEL_EXPERT_ID
from skyrl_train.utils import initialize_ray, get_ray_pg_ready_with_timeout


def _check_gpus(num_gpus: int):
    available = get_available_gpus()
    if len(available) < num_gpus:
        pytest.skip(f"Stage 6 e2e requires >= {num_gpus} GPUs, found {len(available)}: {available}")


def _make_replay_experience(model_config, batch_size=2, seq_len=24, num_actions=8, device="cpu") -> Experience:
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
    return Experience(
        sequences=torch.randint(0, model_config.vocab_size, (B, T), device=device),
        action_log_probs=0.4 * torch.ones((B, num_actions), device=device),
        base_action_log_probs=0.3 * torch.ones((B, num_actions), device=device),
        rollout_logprobs=0.2 * torch.ones((B, num_actions), device=device),
        values=0.5 * torch.ones((B, num_actions), device=device),
        returns=0.5 * torch.ones((B, num_actions), device=device),
        advantages=0.6 * torch.ones((B, num_actions), device=device),
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
        # Leave GPU headroom for the colocated EP-sharded trainer next to the engines.
        cfg.generator.gpu_memory_utilization = 0.45
        # Deterministic sampling so the weight-sync oracle is stable.
        cfg.generator.sampling_params.temperature = 0.0
        cfg.generator.sampling_params.top_p = 1.0
        cfg.generator.sampling_params.top_k = -1

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
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
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

        for result in results:
            assert isinstance(result, dict), "training_step should return a status dict"
            assert "policy_loss" in result and "final_loss" in result
            loss_v = result["final_loss"]
            assert math.isfinite(loss_v), f"loss is not finite: {loss_v}"
            assert "raw_grad_norm" in result, "grad-norm missing — optimizer step did not run"
            gn = result["raw_grad_norm"]
            assert math.isfinite(gn), f"grad-norm is not finite: {gn}"
            assert gn > 0.0, f"grad-norm is zero — no grad flowed through router/experts: {gn}"
        print(
            f"[Stage6] replay+EP+grouped training step: loss={results[0]['final_loss']:.4f} "
            f"grad_norm={results[0]['raw_grad_norm']:.4f} entropy={results[0]['policy_entropy']:.4f}"
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
