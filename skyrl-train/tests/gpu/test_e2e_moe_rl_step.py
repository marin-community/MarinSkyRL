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


def _build_forced_sequences(tokenizer, prompts, num_actions: int, seed: int = 7):
    """Build a FIXED (prompt + forced continuation) token sequence per prompt.

    Returns ``(seqs, num_actions)`` where ``seqs`` is a ``List[List[int]]`` and each
    sequence is ``prompt_ids + forced_continuation`` with the SAME forced
    continuation ids reused by BOTH scoring paths (trainer forward + vLLM
    prompt_logprobs). The continuation is a deterministic pseudo-random run of valid
    token ids (seeded), NOT sampled from the model — the gate scores a fixed target
    sequence, so no sampling is involved and the comparison is fully deterministic.

    The trainer and engine see the IDENTICAL ids; only the model numerics differ.
    """
    rng = random.Random(seed)
    vocab_size = int(getattr(tokenizer, "vocab_size", 32000))
    # Stay well inside the vocab and avoid id 0 / special-token collisions by using a
    # safe interior band; the exact ids are immaterial — both paths score the same band.
    lo, hi = 100, max(101, min(vocab_size - 100, 30000))
    prompt_ids_batch = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, add_special_tokens=False, tokenize=True
    )
    seqs = []
    for pids in prompt_ids_batch:
        cont = [rng.randint(lo, hi) for _ in range(num_actions)]
        seqs.append(list(pids) + cont)
    return seqs, num_actions


def _engine_token_logprobs(prompt_logprobs_batch, seqs, num_actions):
    """Extract per-token chosen-token logprobs from vLLM ``prompt_logprobs``.

    vLLM ``prompt_logprobs[pos]`` is a dict ``{token_id: logprob}`` giving
    ``logP(seq[pos] | seq[:pos])`` (it always includes the actual prompt token even
    when it's outside the requested top-K). The trainer's ``action_log_probs[:, j]``
    (the last ``num_actions`` slots) is ``logP(seq[P+j] | seq[:P+j])`` with
    ``P = len(seq) - num_actions`` (the model_wrapper roll(-1) + ``[-na-1:-1]``
    slice). So trainer action ``j`` aligns with engine prompt position ``P+j``.

    Returns a ``[B, num_actions]`` float tensor (NaN where vLLM omitted a position).
    """
    B = len(seqs)
    out = torch.full((B, num_actions), float("nan"))
    for i in range(B):
        seq = seqs[i]
        P = len(seq) - num_actions
        plp = prompt_logprobs_batch[i] if prompt_logprobs_batch is not None else None
        if plp is None:
            continue
        for j in range(num_actions):
            pos = P + j
            if pos >= len(plp) or plp[pos] is None:
                continue
            tok = seq[pos]
            lp = plp[pos].get(tok)
            if lp is not None:
                out[i, j] = float(lp)
    return out


def test_e2e_moe_rl_step_replay_ep_grouped():
    """One full RL training step on a small MoE under EP=2 + grouped-GEMM +
    router-replay, then a weight-sync round-trip into the EP inference engine.

    Asserts:
      * training step completes; loss + grad-norm finite (no NaN/inf), grad-norm > 0
        (grad flows through router + experts);
      * DETERMINISTIC logprob-agreement gate: the per-token logprobs of a FIXED
        (prompt + forced continuation) sequence, computed by (a) the trainer's
        POST-STEP forward and (b) the synced EP=2 vLLM engine's ``prompt_logprobs``,
        agree within a tight tolerance. This passes iff the real GRPO update was
        faithfully broadcast/resharded/remapped into the inference engine. It
        replaces the prior greedy-decode ``are_responses_similar`` oracle, which was
        a known artifact (the first AdamW step shifts ``router.gate`` ~1e-6
        coherently -> flips greedy argmax -> full Levenshtein divergence -> 0/4),
        and unlike the rejected ``lr=0.0`` no-op it validates a REAL update.
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
        # FIXED (prompt + forced continuation) sequences for the deterministic
        # logprob-agreement gate. Built ONCE; the SAME ids are scored by both the
        # trainer's post-step forward and the synced EP vLLM engine. No sampling.
        SCORE_NUM_ACTIONS = 24
        forced_seqs, score_num_actions = _build_forced_sequences(
            tokenizer, prompts, num_actions=SCORE_NUM_ACTIONS
        )

        # Smoke the engine once so a dead-engine failure surfaces before the (much
        # longer) training step; this generate is NOT the gate (the gate is the
        # post-step logprob agreement below).
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

        # --- (a) TRAINER post-step per-token logprobs on the FIXED sequences. ---
        # Run a NO-grad forward of the just-updated policy on the forced sequences
        # (natural routing — rollout_routed_experts is NOT passed here, matching the
        # engine's native routing) BEFORE offload/weight-sync, so we score the SAME
        # post-step weights the engine will receive. `forward` -> model_wrapper
        # returns `action_log_probs[:, -num_actions:]` = per-token logP of the realized
        # next token at temperature=1.0 (the gate's training-forward temperature).
        max_len = max(len(s) for s in forced_seqs)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # Left-pad so the response slice `[-num_actions-1:-1]` lands on the forced
        # continuation for every row (the model_wrapper slices from the right).
        seq_tensor = torch.full((len(forced_seqs), max_len), pad_id, dtype=torch.long)
        attn_tensor = torch.zeros((len(forced_seqs), max_len), dtype=torch.long)
        for i, s in enumerate(forced_seqs):
            seq_tensor[i, max_len - len(s):] = torch.tensor(s, dtype=torch.long)
            attn_tensor[i, max_len - len(s):] = 1
        # get_model_logits_from_actor sets response_length internally to seq_len-5,
        # so call the worker forward directly with our exact score_num_actions.
        from skyrl_train.training_batch import TrainingInputBatch
        from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch

        fwd_data = TrainingInputBatch({"sequences": seq_tensor, "attention_mask": attn_tensor})
        fwd_data.metadata = {"response_length": score_num_actions}
        fwd_refs = policy.async_run_ray_method("mesh", "forward", fwd_data)
        fwd_out = concatenate_outputs_after_mesh_dispatch(policy.actor_infos, ray.get(fwd_refs))
        trainer_logprobs = fwd_out["output"].float()  # [B, score_num_actions]
        assert trainer_logprobs.shape == (len(forced_seqs), score_num_actions), (
            f"trainer logprob shape {tuple(trainer_logprobs.shape)} != "
            f"{(len(forced_seqs), score_num_actions)}"
        )
        assert torch.isfinite(trainer_logprobs).all(), "trainer post-step logprobs contain non-finite values"

        # --- Weight-sync round-trip into the EP inference engine (G4-4 oracle). ---
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.wake_up(tags=["weights"]))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        policy.offload_to_cpu()
        asyncio.run(client.wake_up(tags=["kv_cache"]))
        asyncio.run(client.reset_prefix_cache())

        # --- (b) ENGINE post-sync per-token logprobs on the SAME FIXED sequences. ---
        # NO sampling: feed the full forced sequence as `prompt_token_ids` and read
        # vLLM `prompt_logprobs` (the exact teacher-scoring path in
        # teacher_engine_client). `max_tokens=1` is required (vLLM must emit >=1
        # token) but the generated token is irrelevant — we only read the prompt
        # logprobs over the forced sequence, so the comparison is fully deterministic.
        score_sampling_params = {
            "max_tokens": 1,
            # prompt_logprobs=1 (the colocated EP engine's max_logprobs cap): vLLM
            # ALWAYS includes the ACTUAL prompt token's logprob in prompt_logprobs[pos]
            # even when it falls outside the requested top-K, and _engine_token_logprobs
            # reads exactly that (plp[pos].get(tok), indexed by the forced token id —
            # NOT a top-K rank). So K=1 yields every forced-token logprob the gate needs
            # while staying within the engine's max_logprobs=1 (no engine/a3 change).
            "prompt_logprobs": 1,  # vLLM ALSO includes the actual prompt token
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
        }
        score_out = asyncio.run(
            client.generate(
                InferenceEngineInput(prompt_token_ids=forced_seqs, sampling_params=score_sampling_params)
            )
        )
        engine_plp = score_out.get("prompt_logprobs")
        assert engine_plp is not None, (
            "EP vLLM engine returned no prompt_logprobs — cannot run the logprob-agreement gate. "
            "(SamplingParams(prompt_logprobs=K) is set; this should not happen on vLLM.)"
        )
        engine_logprobs = _engine_token_logprobs(engine_plp, forced_seqs, score_num_actions)

        # --- THE GATE: numerical agreement of post-step trainer vs synced-engine
        # per-token logprobs over the FIXED forced sequences. ---
        # This passes IFF the post-step weights are faithfully in the EP engine. A
        # mis-propagated update (wrong reshard/remap of router.gate or expert
        # w1/w2/w3) shifts the logits by O(1)+ -> logprobs diverge by >> the tolerance.
        # A faithful sync agrees up to fp/bf16 + grouped-GEMM-vs-HF + EP-reduction-order
        # noise, which on these logprobs is a few e-2 to e-1 — orders of magnitude
        # smaller than a real corruption. We assert on the per-token ABS diff,
        # ignoring any positions vLLM omitted (NaN) and clamping the trainer logprobs
        # to the same finite set.
        valid = torch.isfinite(engine_logprobs) & torch.isfinite(trainer_logprobs)
        n_valid = int(valid.sum().item())
        assert n_valid >= score_num_actions, (
            f"too few comparable logprob positions ({n_valid}); vLLM omitted most prompt positions"
        )
        abs_diff = (trainer_logprobs - engine_logprobs).abs()
        abs_diff_valid = abs_diff[valid]
        max_abs = float(abs_diff_valid.max().item())
        mean_abs = float(abs_diff_valid.mean().item())

        for i in range(len(forced_seqs)):
            row_valid = valid[i]
            t = trainer_logprobs[i][row_valid].tolist()
            e = engine_logprobs[i][row_valid].tolist()
            d = abs_diff[i][row_valid].tolist()
            print(
                f"[Stage6][logprob-gate][seq{i}] n={int(row_valid.sum())} "
                f"max_abs={max(d) if d else float('nan'):.4f} mean_abs={(sum(d)/len(d)) if d else float('nan'):.4f}\n"
                f"  TRAINER: {[round(x, 3) for x in t]}\n"
                f"  ENGINE : {[round(x, 3) for x in e]}\n"
                f"  ABSDIFF: {[round(x, 3) for x in d]}"
            )
        # Tolerance justification: identical fp32 logprobs are unattainable across the
        # FSDP2 bf16 grouped-GEMM trainer forward and vLLM's native bf16 MoE forward
        # (different kernels, expert-GEMM grouping, EP all-reduce order, RoPE/attn
        # impl). Empirically that residual is a few e-2 per token on logprobs of this
        # magnitude. A MIS-PROPAGATED update is not subtle: a wrong router.gate or
        # expert tensor flips top-k expert selection and/or shifts logits by O(1+),
        # giving per-token diffs of >= 1.0 (and frequently >> that). MAX_ABS_TOL=0.5
        # sits comfortably between the two regimes: it tolerates the full fp/kernel
        # noise band with margin while still catching any genuine transport corruption
        # by an order of magnitude. MEAN_ABS_TOL=0.15 guards against a small-but-
        # systematic offset (a partial/misaligned remap) that a max-only check could
        # miss. Both thresholds are TEST-ONLY; the a3 path (ep_size=1) is untouched.
        MAX_ABS_TOL = 0.5
        MEAN_ABS_TOL = 0.15
        print(
            f"[Stage6][logprob-gate] OVERALL max_abs={max_abs:.4f} (tol={MAX_ABS_TOL}) "
            f"mean_abs={mean_abs:.4f} (tol={MEAN_ABS_TOL}) over {n_valid} tokens"
        )
        assert max_abs <= MAX_ABS_TOL, (
            f"weight-sync round-trip MIS-PROPAGATED the post-step update: max per-token logprob "
            f"diff {max_abs:.4f} > {MAX_ABS_TOL} (mean {mean_abs:.4f}). Trainer post-step and synced "
            f"EP-engine logprobs disagree by orders of magnitude -> the broadcast/reshard/remap did "
            f"not faithfully land the real GRPO update in the inference engine."
        )
        assert mean_abs <= MEAN_ABS_TOL, (
            f"weight-sync round-trip has a systematic logprob offset: mean per-token diff {mean_abs:.4f} "
            f"> {MEAN_ABS_TOL} (max {max_abs:.4f}) -> a partial/misaligned remap of the post-step weights."
        )
        print(
            f"[Stage6] weight-sync logprob-agreement gate PASS: post-step trainer weights are "
            f"faithfully in the EP engine (max_abs={max_abs:.4f}, mean_abs={mean_abs:.4f})"
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
