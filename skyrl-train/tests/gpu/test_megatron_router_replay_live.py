"""Capture real vLLM Grug routes and replay them through a Megatron update.

Run the opt-in module on four otherwise idle Hopper-or-newer GPUs from the
checkout root with ``bash skyrl-train/tests/gpu/run_live_replay_iris.sh``.
The first test uses two GPUs; the EP2 test uses all four. The fixture uses
Grug's gated norms, XSA, query bias, shared experts, and top-4 MoE.
"""

import asyncio
import json
import math

import pytest
import ray
import torch

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import (
    LM_HEAD_NAME,
    ROUTER_NAME,
    STACKED_EXPERT_NAME,
    assert_engine_weights,
    grug_engine_client,
    rank0_validation_snapshot,
    rollout_training_batch,
)
from tests.gpu.test_grug_megatron import (
    BIAS_NAMES,
    NUM_LAYERS,
    SERVING_EXPERT_INDEX_BY_NAME,
    _config,
    _init_policy,
    _megatron_response_logprobs,
    _train_step,
    _write_tiny_checkpoint,
)

TOPK = 4
RESPONSE_LENGTH = 4
PROMPTS = (
    (1, 17, 29, 5, 11, 3),
    (1, 19, 31, 7, 13, 3),
    (1, 23, 37, 11, 17, 3),
    (1, 31, 41, 13, 19, 3),
)


def _scores(policy, batch: TrainingInputBatch, routes: torch.Tensor) -> torch.Tensor:
    batch["rollout_routed_experts"] = routes
    return _megatron_response_logprobs(policy, batch)


@pytest.mark.vllm
def test_actual_vllm_routes_reach_megatron_scoring_and_update(tmp_path) -> None:
    require_hoppers(2)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_checkpoint(model_path, num_experts_per_tok=TOPK)
    cfg = _config(str(model_path), world_size=1, pp=1, ep=1)
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    cfg.trainer.policy.grug_query_bias_update_mode = "loss_free"
    cfg.trainer.policy.grug_query_bias_update_rate = 0.001
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    initialize_ray(cfg)
    client = grug_engine_client(cfg, str(model_path), capture_routes=True)
    try:
        policy = _init_policy(cfg, 1)
        names = [LM_HEAD_NAME, ROUTER_NAME, STACKED_EXPERT_NAME, *BIAS_NAMES]
        before = rank0_validation_snapshot(policy, names)
        # The Megatron bridge rounds BF16 parameters on import, whereas the
        # initial vLLM HF load can retain an FP32 router. Sync first so the
        # rollout and training comparison uses identical effective weights.
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        assert_engine_weights(
            client,
            [LM_HEAD_NAME, ROUTER_NAME, *BIAS_NAMES, *SERVING_EXPERT_INDEX_BY_NAME],
            before,
            BIAS_NAMES,
            SERVING_EXPERT_INDEX_BY_NAME,
        )

        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        sampling_params.update({"temperature": 0.0, "max_tokens": RESPONSE_LENGTH, "ignore_eos": True, "logprobs": 1})
        rollout = asyncio.run(
            client.generate(
                InferenceEngineInput(
                    prompt_token_ids=[list(prompt) for prompt in PROMPTS], sampling_params=sampling_params
                )
            )
        )
        assert all(len(tokens) == RESPONSE_LENGTH for tokens in rollout["response_ids"])
        captured = torch.tensor(rollout["routed_experts"], dtype=torch.int32)
        assert captured.shape == (len(PROMPTS), RESPONSE_LENGTH, NUM_LAYERS, TOPK)
        assert torch.all((captured >= 0) & (captured < 8))

        batch = rollout_training_batch([list(prompt) for prompt in PROMPTS], rollout)
        native = _scores(policy, batch, torch.zeros_like(captured))
        replayed = _scores(policy, batch, captured)
        rollout_scores = torch.tensor(rollout["response_logprobs"], dtype=torch.float32)
        valid = batch["response_mask"].bool()
        native_gap = (native - rollout_scores)[valid].abs()
        replay_gap = (replayed - rollout_scores)[valid].abs()
        print(
            "LIVE_ROUTE_REPLAY="
            + json.dumps(
                {
                    "model": "feature-complete random Grug, 8 layers, 8 experts, top-4",
                    "layout": "vLLM TP1/EP1; Megatron TP1/PP1/EP1/CP1",
                    "captured_shape": list(captured.shape),
                    "native_logprob_max_abs": native_gap.max().item(),
                    "replay_logprob_max_abs": replay_gap.max().item(),
                    "native_ratio_max_deviation": (torch.exp(native - rollout_scores)[valid] - 1).abs().max().item(),
                    "replay_ratio_max_deviation": (torch.exp(replayed - rollout_scores)[valid] - 1).abs().max().item(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        batch["rollout_routed_experts"] = captured
        status = _train_step(policy, batch)
        assert status["router_replay/hit_fraction"] == 1.0, status
        assert status["router_replay/executed_route_match_fraction"] == 1.0, status
        assert status["router_replay/router_grad_norm"] > 0.0, status
        assert status["router_replay/query_bias_max_change"] > 0.0, status
        assert math.isfinite(status["policy_loss"]), status

        after = rank0_validation_snapshot(policy, names)
        assert not torch.equal(after[ROUTER_NAME], before[ROUTER_NAME])
        print(
            "LIVE_ROUTE_UPDATE="
            + json.dumps(
                {
                    "router_grad_norm": status["router_replay/router_grad_norm"],
                    "router_weight_max_change": (after[ROUTER_NAME] - before[ROUTER_NAME]).abs().max().item(),
                    "replay_hit_fraction": status["router_replay/hit_fraction"],
                    "replay_sentinel_fraction": status["router_replay/sentinel_fraction"],
                    "native_route_mismatch_fraction": status["router_replay/native_mismatch_fraction"],
                    "native_expert_set_mismatch_fraction": status["router_replay/native_set_mismatch_fraction"],
                    "executed_route_match_fraction": status["router_replay/executed_route_match_fraction"],
                    "query_bias_max_change": status["router_replay/query_bias_max_change"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        assert any(not torch.equal(after[name], before[name]) for name in BIAS_NAMES)
        for name in BIAS_NAMES:
            assert torch.isfinite(after[name]).all()
            assert after[name].float().mean().abs() < 1e-6
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        assert_engine_weights(
            client,
            [LM_HEAD_NAME, ROUTER_NAME, *BIAS_NAMES, *SERVING_EXPERT_INDEX_BY_NAME],
            after,
            BIAS_NAMES,
            SERVING_EXPERT_INDEX_BY_NAME,
        )
    finally:
        ray.shutdown()

    # A fresh flag-off policy scores the exact same rollout tokens with the
    # same imported weights. The sentinel pass above is an internal native
    # control; this second actor checks the public flag-off path itself.
    cfg.trainer.policy.fsdp_config.moe_router_replay = False
    cfg.trainer.policy.grug_query_bias_update_mode = "frozen"
    cfg.trainer.policy.grug_query_bias_update_rate = None
    initialize_ray(cfg)
    try:
        off_policy = _init_policy(cfg, 1)
        off_weights = rank0_validation_snapshot(off_policy, names)
        for name in names:
            torch.testing.assert_close(off_weights[name], before[name], rtol=0, atol=0)
        off_batch = rollout_training_batch([list(prompt) for prompt in PROMPTS], rollout)
        off_scores = _megatron_response_logprobs(off_policy, off_batch)
        off_gap = (off_scores - rollout_scores)[valid].abs()
        flag_off_vs_sentinel = (off_scores - native)[valid].abs().max().item()
        assert flag_off_vs_sentinel < 1e-6, flag_off_vs_sentinel
        print(
            "LIVE_ROUTE_FLAG_OFF="
            + json.dumps(
                {
                    "logprob_max_abs": off_gap.max().item(),
                    "ratio_max_deviation": (torch.exp(off_scores - rollout_scores)[valid] - 1).abs().max().item(),
                    "flag_off_vs_sentinel_max_abs": flag_off_vs_sentinel,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        ray.shutdown()


@pytest.mark.vllm
def test_actual_vllm_routes_reach_ep2_megatron_training(tmp_path) -> None:
    require_hoppers(4)
    model_path = tmp_path / "ep2-model"
    model_path.mkdir()
    _write_tiny_checkpoint(model_path, num_experts_per_tok=TOPK)
    cfg = _config(str(model_path), world_size=2, pp=1, ep=2)
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    cfg.trainer.policy.grug_query_bias_update_mode = "loss_free"
    cfg.trainer.policy.grug_query_bias_update_rate = 0.001
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 2
    initialize_ray(cfg)
    client = grug_engine_client(cfg, str(model_path), capture_routes=True)
    try:
        policy = _init_policy(cfg, 2)
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        sampling_params.update({"temperature": 0.0, "max_tokens": RESPONSE_LENGTH, "ignore_eos": True, "logprobs": 1})
        rollout = asyncio.run(
            client.generate(
                InferenceEngineInput(
                    prompt_token_ids=[list(prompt) for prompt in PROMPTS], sampling_params=sampling_params
                )
            )
        )
        captured = torch.tensor(rollout["routed_experts"], dtype=torch.int32)
        assert captured.shape == (len(PROMPTS), RESPONSE_LENGTH, NUM_LAYERS, TOPK)
        batch = rollout_training_batch([list(prompt) for prompt in PROMPTS], rollout)
        batch["rollout_routed_experts"] = captured
        replayed = _megatron_response_logprobs(policy, batch)
        rollout_scores = torch.tensor(rollout["response_logprobs"], dtype=torch.float32)
        valid = batch["response_mask"].bool()
        score_gap = (replayed - rollout_scores)[valid].abs().max().item()
        status = _train_step(policy, batch)
        assert status["router_replay/hit_fraction"] == 1.0, status
        assert status["router_replay/executed_route_match_fraction"] == 1.0, status
        assert status["router_replay/router_grad_norm"] > 0, status
        assert status["router_replay/query_bias_max_change"] > 0, status
        print(
            "LIVE_ROUTE_EP2="
            + json.dumps(
                {
                    "model": "feature-complete random Grug, 8 layers, 8 experts, top-4",
                    "layout": "vLLM TP1/EP2; Megatron TP1/PP1/EP2/CP1",
                    "captured_shape": list(captured.shape),
                    "replay_logprob_max_abs": score_gap,
                    "executed_route_match_fraction": status["router_replay/executed_route_match_fraction"],
                    "router_grad_norm": status["router_replay/router_grad_norm"],
                    "query_bias_max_change": status["router_replay/query_bias_max_change"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        ray.shutdown()
