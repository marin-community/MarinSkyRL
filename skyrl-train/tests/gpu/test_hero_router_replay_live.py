"""Replay actual vLLM expert IDs through a feature-complete tiny Hero update."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
import pytest
import ray
import torch

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.models.grug_moe import GrugMoeConfig
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
    trained_uri = os.environ.get("HERO_REPLAY_TRAINED_URI")
    policy_world_size = 4 if trained_uri else 1
    require_hoppers(policy_world_size + 1)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model_path = tmp_path / "hero"
    model_path.mkdir()
    staged_bytes = _stage_trained_checkpoint(trained_uri, model_path) if trained_uri else 0
    if not trained_uri:
        write_tiny_hero_checkpoint(model_path)
    model_config = GrugMoeConfig.from_pretrained(model_path)
    cfg = _config(str(model_path), world_size=policy_world_size, pp=1, ep=policy_world_size)
    if trained_uri:
        cfg.trainer.policy.optimizer_config.lr = 1e-4
        # Four policy ranks share four examples, so each rank has one example.
        cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    cfg.trainer.policy.grug_query_bias_update_mode = "loss_free"
    cfg.trainer.policy.grug_query_bias_update_rate = 0.001
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    prompts = [[1, 17 + row, 29, 5, 11, 3] for row in range(4)]
    bias_names = [f"model.layers.{layer}.mlp.router.bias" for layer in range(model_config.num_hidden_layers)]
    names = ["model.layers.0.mlp.router.weight", *bias_names]
    if not trained_uri:
        names.insert(0, "lm_head.weight")
    sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    sampling.update(temperature=0.0, max_tokens=4, ignore_eos=True, logprobs=1)

    initialize_ray(cfg)
    try:
        client = grug_engine_client(cfg, str(model_path), capture_routes=True, enable_flashinfer_autotune=False)
        policy = _init_policy(cfg, policy_world_size)
        before = rank0_validation_snapshot(policy, names)
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        assert_engine_weights(client, names, before, bias_names, {})

        rollout = asyncio.run(client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling)))
        assert all(len(tokens) == 4 for tokens in rollout["response_ids"])
        captured = torch.tensor(rollout["routed_experts"], dtype=torch.int32)
        assert captured.shape == (4, 4, model_config.num_hidden_layers, model_config.num_experts_per_tok)
        assert torch.all((captured >= 0) & (captured < model_config.num_local_experts))
        batch = rollout_training_batch(prompts, rollout)
        native = _score(policy, batch, torch.zeros_like(captured))
        replayed = _score(policy, batch, captured)
        serving = torch.tensor(rollout["response_logprobs"], dtype=torch.float32)
        valid = batch["response_mask"].bool()
        batch["action_log_probs"] = replayed.float()
        status = _train_step(policy, batch)
        train_diagnostic = {
            "phase": "after_train",
            "model": trained_uri or "random Hero schema-v2",
            "native_logprob_max_abs": (native - serving)[valid].abs().max().item(),
            "replay_logprob_max_abs": (replayed - serving)[valid].abs().max().item(),
            "status": {
                key: float(value)
                for key, value in status.items()
                if key.startswith("router_replay/")
                or key in {"raw_grad_norm", "policy_update_steps", "log_ratio_abs_max"}
            },
        }
        print("LIVE_HERO_REPLAY_TRAIN_STATUS=" + json.dumps(train_diagnostic, sort_keys=True), flush=True)
        result_uri = os.environ.get("HERO_REPLAY_RESULT_URI")
        if result_uri:
            _put_s3_json(result_uri, train_diagnostic)
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
            assert (after[name].float().mean() - before[name].float().mean()).abs() < 1e-6
        on_metrics = {
            "model": trained_uri or "random Hero schema-v2, 4 layers, 16 experts, top-8, latent MoE, ShortConv",
            "model_layers": model_config.num_hidden_layers,
            "model_experts": model_config.num_local_experts,
            "staged_checkpoint_bytes": staged_bytes,
            "layout": f"vLLM TP1/EP1; Megatron TP1/PP1/EP{policy_world_size}/CP1",
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
        off_policy = _init_policy(cfg, policy_world_size)
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
        if result_uri:
            _put_s3_json(result_uri, on_metrics)
    finally:
        ray.shutdown()


def _score(policy, batch, routes: torch.Tensor) -> torch.Tensor:
    batch["rollout_routed_experts"] = routes
    return _megatron_response_logprobs(policy, batch)


def _s3_target(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"expected an s3://bucket/key URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _s3_client():
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ["AWS_ENDPOINT_URL"]
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="auto",
        config=Config(s3={"addressing_style": "virtual"}, retries={"max_attempts": 5}),
    )


def _stage_trained_checkpoint(uri: str, destination: Path) -> int:
    bucket, prefix = _s3_target(uri)
    prefix = prefix.rstrip("/") + "/"
    for attempt in range(4):
        client = _s3_client()
        listed = [
            item
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for item in page.get("Contents", ())
        ]
        objects = [
            item
            for item in listed
            if item["Key"].removeprefix(prefix)
            in {
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            }
            or item["Key"].removeprefix(prefix).startswith("model-")
            and item["Key"].endswith(".safetensors")
        ]
        filenames = {item["Key"].removeprefix(prefix) for item in objects}
        if {"config.json", "model.safetensors.index.json", "tokenizer.json"} <= filenames:
            break
        if attempt < 3:
            time.sleep(2**attempt)
    assert {"config.json", "model.safetensors.index.json", "tokenizer.json"} <= filenames, (
        f"checkpoint list missing required files after four attempts: listed={len(listed)}, "
        f"sample={[item['Key'] for item in listed[:3]]}, selected={sorted(filenames)[:5]}"
    )
    assert any(name.startswith("model-") and name.endswith(".safetensors") for name in filenames), filenames
    assert all(Path(name).name == name for name in filenames), filenames

    def download(item: dict) -> None:
        name = item["Key"].removeprefix(prefix)
        target = destination / name
        client.download_file(bucket, item["Key"], str(target))
        assert target.stat().st_size == item["Size"], (name, target.stat().st_size, item["Size"])

    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(download, objects))
    return sum(item["Size"] for item in objects)


def _put_s3_json(uri: str, metrics: dict) -> None:
    bucket, key = _s3_target(uri)
    _s3_client().put_object(Bucket=bucket, Key=key, Body=json.dumps(metrics, sort_keys=True).encode())
