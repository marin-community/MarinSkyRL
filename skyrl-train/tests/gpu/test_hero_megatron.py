"""Hero architecture through the actual Megatron worker, with split-expert checkpoints."""

import asyncio
import json
import math
from pathlib import Path

from omegaconf import open_dict
import pytest
import ray
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from skyrl_train.models.grug_moe import GrugMoeConfig
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
    _padded_batch,
    _write_tiny_checkpoint,
)


def _train_step(policy, batch):
    outputs = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))
    status = outputs[0].metadata["train_status"]
    assert math.isfinite(status["policy_loss"])
    assert status["policy_update_steps"] == 1
    assert status["raw_grad_norm"] > 0
    return status


def write_tiny_hero_checkpoint(path: Path):
    """Create all Hero parameter families, retaining an ordinary saved tokenizer."""
    shape = dict(
        hidden_size=128,
        intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_local_experts=16,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        sliding_window=8,
    )
    _write_tiny_checkpoint(path, shape=shape, num_experts_per_tok=8)
    config = GrugMoeConfig.from_pretrained(path)
    state = load_file(str(path / "model.safetensors"))
    torch.manual_seed(23)
    # The borrowed tokenizer has an odd vocabulary; Hero's real vocabulary is
    # divisible by the supported tensor-parallel sizes.
    vocab_size = (config.vocab_size + 127) // 128 * 128
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        state[name] = torch.nn.functional.pad(state[name], (0, 0, 0, vocab_size - config.vocab_size))
    config.vocab_size = vocab_size
    latent = 64
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        for projection, dims in (("gate_proj", (64, latent)), ("up_proj", (64, latent)), ("down_proj", (latent, 64))):
            del state[f"{prefix}.mlp.experts.{projection}.weight"]
            for expert in range(config.num_local_experts):
                state[f"{prefix}.mlp.experts.{expert}.{projection}.weight"] = torch.randn(*dims) * 0.1
        for projection in ("gate_proj", "up_proj", "down_proj"):
            original = state.pop(f"{prefix}.shared_expert.{projection}.weight")
            for expert in range(2):
                state[f"{prefix}.shared_experts.{expert}.{projection}.weight"] = torch.randn_like(original) * 0.1
        state[f"{prefix}.mlp.latent_down_proj.weight"] = torch.randn(latent, config.hidden_size) * 0.1
        state[f"{prefix}.mlp.latent_up_proj.weight"] = torch.randn(config.hidden_size, latent) * 0.1
        state[f"{prefix}.mlp.latent_norm.weight"] = torch.ones(latent)
        for site, channels in (("self_attn.sconv_k", 64), ("sconv_attn", 128), ("sconv_mlp", 128)):
            weight = torch.randn(4, channels) * 0.1
            weight[0] += 1
            state[f"{prefix}.{site}.weight"] = weight
    config = GrugMoeConfig(
        **{
            **config.to_dict(),
            "latent_dim": latent,
            "num_shared_experts": 2,
            "local_kv_heads": 4,
            "global_kv_heads": 2,
            "rope_fused": True,
            "sconv": True,
            "grugmoe_artifact_schema_version": 2,
        }
    )
    config.save_pretrained(path)
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    return state


@pytest.mark.parametrize(
    "tp,pp,ep,cp,packing,overlap_param_gather,optimizer_offload",
    [
        (1, 1, 1, 1, False, False, None),
        (1, 2, 1, 1, False, False, None),
        (1, 1, 2, 1, True, False, None),
        (1, 1, 1, 2, True, False, None),
        (2, 1, 1, 1, True, False, None),
        (1, 1, 1, 2, True, True, None),
        (2, 2, 4, 2, True, True, None),
        pytest.param(1, 1, 2, 2, True, True, 0.0, id="precision-aware-gpu"),
        pytest.param(1, 1, 2, 2, True, True, 0.5, id="half-offloaded-adamw"),
        pytest.param(1, 1, 2, 2, True, True, 1.0, id="cpu-adamw"),
    ],
)
def test_hero_worker_repeated_updates(tmp_path, tp, pp, ep, cp, packing, overlap_param_gather, optimizer_offload):
    world_size = pp * max(tp * cp, ep)
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    original = write_tiny_hero_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=world_size, pp=pp, ep=ep)
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.context_parallel_size = cp
    cfg.trainer.policy.megatron_config.ddp_config.overlap_param_gather = overlap_param_gather
    if optimizer_offload is not None:
        megatron = cfg.trainer.policy.megatron_config
        cfg.trainer.flash_attn = True
        megatron.ddp_config.grad_reduce_in_fp32 = False
        megatron.optimizer_checkpoint_sharding_type = "dp_reshardable"
        with open_dict(megatron.transformer_config_kwargs):
            megatron.transformer_config_kwargs.deterministic_mode = True
        with open_dict(megatron.optimizer_config_kwargs):
            megatron.optimizer_config_kwargs.use_precision_aware_optimizer = True
            megatron.optimizer_config_kwargs.store_param_remainders = False
            megatron.optimizer_config_kwargs.optimizer_cpu_offload = optimizer_offload > 0
            megatron.optimizer_config_kwargs.optimizer_offload_fraction = optimizer_offload
            megatron.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d = False
    cfg.trainer.use_sample_packing = packing
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    batch = _padded_batch(tokenizer.pad_token_id, prompt_length=48, response_length=48, variable_lengths=True)
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg, world_size)
        names = list(original)
        before = rank0_validation_snapshot(policy, names)
        for name in names:
            expected = original[name] if name.endswith("router.bias") else original[name].to(torch.bfloat16).float()
            torch.testing.assert_close(before[name], expected, rtol=0, atol=0)
        selected_names = [
            "model.layers.0.mlp.experts.3.gate_proj.weight",
            "model.layers.0.mlp.experts.15.down_proj.weight",
        ]
        selected = rank0_validation_snapshot(policy, selected_names)
        assert set(selected) == set(selected_names)
        for name in selected_names:
            torch.testing.assert_close(selected[name], before[name], rtol=0, atol=0)
        initial_scores = _megatron_response_logprobs(policy, batch)
        assert torch.isfinite(initial_scores).all()
        repeated_scores = _megatron_response_logprobs(policy, batch)
        torch.testing.assert_close(initial_scores, repeated_scores, rtol=0, atol=0)
        updated_names = [
            "model.layers.0.mlp.latent_down_proj.weight",
            "model.layers.0.sconv_mlp.weight",
            "model.layers.0.self_attn.sconv_k.weight",
            "model.layers.0.shared_experts.1.up_proj.weight",
        ]
        previous_step = before
        metrics = []
        for _ in range(3):
            scores = _megatron_response_logprobs(policy, batch)
            batch["action_log_probs"] = (scores * batch["response_mask"]).float()
            status = _train_step(policy, batch)
            assert status["log_ratio_abs_max"] < 1e-3, status
            metrics.append(status)
            updated = rank0_validation_snapshot(policy, updated_names)
            for name in updated_names:
                assert not torch.equal(previous_step[name], updated[name]), name
            previous_step = updated
        after = rank0_validation_snapshot(policy, names)
        for name in names:
            if name.endswith("router.bias"):
                torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)
            assert torch.isfinite(after[name]).all(), name
        final_scores = _megatron_response_logprobs(policy, batch)
        assert torch.isfinite(final_scores).all()
        assert not torch.equal(initial_scores, final_scores)
        batch["action_log_probs"] = (final_scores * batch["response_mask"]).float()
        checkpoint = str(tmp_path / "checkpoint")
        ray.get(
            policy.async_run_ray_method("pass_through", "save_checkpoint", ckpt_dir=checkpoint, tokenizer=tokenizer)
        )
        continued = []
        for _ in range(2):
            _train_step(policy, batch)
            continued.append(rank0_validation_snapshot(policy, names))
        ray.get(policy.async_run_ray_method("pass_through", "load_checkpoint", ckpt_dir=checkpoint))
        restored = rank0_validation_snapshot(policy, names)
        for name in names:
            torch.testing.assert_close(restored[name], after[name], rtol=0, atol=0)
        restored_scores = _megatron_response_logprobs(policy, batch)
        torch.testing.assert_close(restored_scores, final_scores, rtol=0, atol=0)
        for expected in continued:
            _train_step(policy, batch)
            resumed = rank0_validation_snapshot(policy, names)
            for name in names:
                torch.testing.assert_close(resumed[name], expected[name], rtol=0, atol=0)
        print(json.dumps({"tp": tp, "pp": pp, "ep": ep, "cp": cp, "packing": packing, "metrics": metrics}, default=str))
    finally:
        ray.shutdown()


@pytest.mark.vllm
def test_hero_replay_checkpoint_and_weight_publication(tmp_path, monkeypatch):
    require_hoppers(2)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_hero_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=1, pp=1, ep=1)
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bias_names = [f"model.layers.{layer}.mlp.router.bias" for layer in range(4)]
    names = [
        "lm_head.weight",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.latent_down_proj.weight",
        "model.layers.0.shared_experts.1.up_proj.weight",
        "model.layers.0.self_attn.sconv_k.weight",
        "model.layers.0.sconv_attn.weight",
        "model.layers.0.sconv_mlp.weight",
        *bias_names,
    ]
    prompts = [[1, 17 + row, 29, 5, 11, 3] for row in range(4)]
    sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    sampling.update(temperature=0.0, max_tokens=4, ignore_eos=True, logprobs=1)
    request = InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling)
    initialize_ray(cfg)
    try:
        # vLLM's autotune dummy forwards omit ShortConv cache metadata.
        client = grug_engine_client(cfg, str(model_path), capture_routes=True, enable_flashinfer_autotune=False)
        policy = _init_policy(cfg, 1)
        before = rank0_validation_snapshot(policy, names)
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        assert_engine_weights(client, names, before, bias_names, {})
        rollout = asyncio.run(client.generate(request))
        captured = torch.tensor(rollout["routed_experts"], dtype=torch.int32)
        assert captured.shape == (4, 4, 4, 8)
        assert torch.all((captured >= 0) & (captured < 16))
        batch = rollout_training_batch(prompts, rollout)
        batch["rollout_routed_experts"] = captured
        scores = _megatron_response_logprobs(policy, batch)
        serving_scores = torch.tensor(rollout["response_logprobs"])
        assert torch.isfinite(scores).all()
        # Keep the independent backend gap visible; replay execution itself is
        # checked below and is not a claim of numerical parity with Levanter.
        gap = (scores - serving_scores).abs()
        batch["action_log_probs"] = scores.float()
        status = _train_step(policy, batch)
        assert status["router_replay/hit_fraction"] == 1.0
        assert status["router_replay/executed_route_match_fraction"] == 1.0
        assert status["router_replay/router_grad_norm"] > 0.0
        after = rank0_validation_snapshot(policy, names)
        for name in bias_names:
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)
        assert not torch.equal(after["lm_head.weight"], before["lm_head.weight"])
        checkpoint = str(tmp_path / "checkpoint")
        ray.get(
            policy.async_run_ray_method("pass_through", "save_checkpoint", ckpt_dir=checkpoint, tokenizer=tokenizer)
        )
        batch["action_log_probs"] = _megatron_response_logprobs(policy, batch).float()
        _train_step(policy, batch)
        ray.get(policy.async_run_ray_method("pass_through", "load_checkpoint", ckpt_dir=checkpoint))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        assert_engine_weights(client, names, after, bias_names, {})
        updated = asyncio.run(client.generate(request))
        assert all(len(tokens) == 4 for tokens in updated["response_ids"])
        assert torch.isfinite(torch.tensor(updated["response_logprobs"])).all()
        print(
            json.dumps(
                {
                    "hero_replay": status,
                    "vllm_logprob_gap_max": gap.max().item(),
                    "vllm_logprob_gap_mean": gap.mean().item(),
                }
            )
        )
    finally:
        ray.shutdown()
