"""Replay actual vLLM expert IDs through a feature-complete tiny Hero update."""

import asyncio
import json

import pytest
import ray
import torch

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import (
    assert_engine_weights,
    grug_engine_client,
    rank0_validation_snapshot,
    rollout_training_batch,
)
from tests.gpu.test_grug_megatron import (
    _config,
    _init_policy,
    _megatron_response_logprobs,
    _train_step,
)
from tests.gpu.test_hero_megatron import write_tiny_hero_checkpoint


@pytest.mark.vllm
def test_live_hero_routes_survive_recompute_and_update(tmp_path, monkeypatch) -> None:
    require_hoppers(2)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model_path = tmp_path / "hero"
    model_path.mkdir()
    write_tiny_hero_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=1, pp=1, ep=1)
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    cfg.trainer.policy.grug_query_bias_update_mode = "loss_free"
    cfg.trainer.policy.grug_query_bias_update_rate = 0.001
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    prompts = [[1, 17 + row, 29, 5, 11, 3] for row in range(4)]
    bias_names = [f"model.layers.{layer}.mlp.router.bias" for layer in range(4)]
    names = ["lm_head.weight", "model.layers.0.mlp.router.weight", *bias_names]
    sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    sampling.update(temperature=0.0, max_tokens=4, ignore_eos=True, logprobs=1)

    initialize_ray(cfg)
    try:
        client = grug_engine_client(cfg, str(model_path), capture_routes=True, enable_flashinfer_autotune=False)
        policy = _init_policy(cfg, 1)
        before = rank0_validation_snapshot(policy, names)
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        assert_engine_weights(client, names, before, bias_names, {})

        rollout = asyncio.run(client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling)))
        assert all(len(tokens) == 4 for tokens in rollout["response_ids"])
        captured = torch.tensor(rollout["routed_experts"], dtype=torch.int32)
        assert captured.shape == (4, 4, 4, 8)
        assert torch.all((captured >= 0) & (captured < 16))
        batch = rollout_training_batch(prompts, rollout)
        native = _score(policy, batch, torch.zeros_like(captured))
        replayed = _score(policy, batch, captured)
        serving = torch.tensor(rollout["response_logprobs"], dtype=torch.float32)
        valid = batch["response_mask"].bool()
        batch["action_log_probs"] = replayed.float()
        status = _train_step(policy, batch)
        assert status["router_replay/hit_fraction"] == 1.0, status
        assert status["router_replay/executed_route_match_fraction"] == 1.0, status
        assert status["router_replay/router_grad_norm"] > 0.0, status
        assert status["router_replay/query_bias_max_change"] > 0.0, status
        assert status["log_ratio_abs_max"] < 1e-3, status
        after = rank0_validation_snapshot(policy, names)
        assert not torch.equal(after["model.layers.0.mlp.router.weight"], before["model.layers.0.mlp.router.weight"])
        assert any(not torch.equal(after[name], before[name]) for name in bias_names)
        for name in bias_names:
            assert torch.isfinite(after[name]).all()
            assert after[name].float().mean().abs() < 1e-6
        on_metrics = {
            "model": "random Hero schema-v2, 4 layers, 16 experts, top-8, latent MoE, ShortConv",
            "layout": "vLLM TP1/EP1; Megatron TP1/PP1/EP1/CP1",
            "captured_shape": list(captured.shape),
            "native_logprob_max_abs": (native - serving)[valid].abs().max().item(),
            "replay_logprob_max_abs": (replayed - serving)[valid].abs().max().item(),
            "native_ratio_max_deviation": (torch.exp(native - serving)[valid] - 1).abs().max().item(),
            "replay_ratio_max_deviation": (torch.exp(replayed - serving)[valid] - 1).abs().max().item(),
            "executed_route_match_fraction": status["router_replay/executed_route_match_fraction"],
            "native_expert_set_mismatch_fraction": status["router_replay/native_set_mismatch_fraction"],
            "router_grad_norm": status["router_replay/router_grad_norm"],
            "query_bias_max_change": status["router_replay/query_bias_max_change"],
        }
    finally:
        ray.shutdown()

    cfg.trainer.policy.fsdp_config.moe_router_replay = False
    cfg.trainer.policy.grug_query_bias_update_mode = "frozen"
    cfg.trainer.policy.grug_query_bias_update_rate = None
    initialize_ray(cfg)
    try:
        off_policy = _init_policy(cfg, 1)
        off_weights = rank0_validation_snapshot(off_policy, names)
        for name in names:
            torch.testing.assert_close(off_weights[name], before[name], rtol=0, atol=0)
        off_batch = rollout_training_batch(prompts, rollout)
        off_scores = _megatron_response_logprobs(off_policy, off_batch)
        flag_off_vs_sentinel = (off_scores - native)[valid].abs().max().item()
        assert flag_off_vs_sentinel < 1e-6, flag_off_vs_sentinel
        on_metrics["flag_off_vs_sentinel_max_abs"] = flag_off_vs_sentinel
        on_metrics["flag_off_ratio_max_deviation"] = (torch.exp(off_scores - serving)[valid] - 1).abs().max().item()
        print("LIVE_HERO_ROUTE_REPLAY=" + json.dumps(on_metrics, sort_keys=True), flush=True)
    finally:
        ray.shutdown()


def _score(policy, batch, routes: torch.Tensor) -> torch.Tensor:
    batch["rollout_routed_experts"] = routes
    return _megatron_response_logprobs(policy, batch)
