"""Replay actual vLLM expert IDs through tiny or trained Hero updates."""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
import numpy as np
import pytest
import ray
import torch
from safetensors import safe_open
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.models.grug_moe import GrugMoeConfig
from skyrl_train.trajectory_runners.trajectory_processing import (
    align_full_prefix_routes_to_trainer,
    extract_full_prefix_routes_from_rollout_details,
)
from skyrl_train.utils import initialize_ray

from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import (
    assert_engine_weights,
    grug_engine_client,
    rank0_validation_snapshot,
    rollout_training_batch,
)
from tests.gpu.hero_replay_checkpoint import split_stacked_hero_checkpoint
from tests.gpu.test_grug_megatron import (
    _config,
    _init_policy,
    _megatron_response_logprobs,
)
from tests.gpu.test_hero_megatron import _train_step, write_tiny_hero_checkpoint


@pytest.mark.vllm
def test_live_hero_routes_survive_recompute_and_update(tmp_path, monkeypatch) -> None:
    trained_uri = os.environ.get("HERO_REPLAY_TRAINED_URI")
    policy_world_size = 4 if trained_uri else 1
    require_hoppers(policy_world_size + 1)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model_path = tmp_path / "hero"
    staged_bytes = 0
    split_tensors = 0
    if trained_uri:
        stacked_path = tmp_path / "hero-source"
        stacked_path.mkdir()
        staged_bytes = _stage_trained_checkpoint(trained_uri, stacked_path)
        split_tensors = split_stacked_hero_checkpoint(stacked_path, model_path)
        assert split_tensors > 0, "trained export unexpectedly had no stacked experts to split"
    else:
        model_path.mkdir()
        write_tiny_hero_checkpoint(model_path)
    model_config = GrugMoeConfig.from_pretrained(model_path)
    cfg = _config(str(model_path), world_size=policy_world_size, pp=1, ep=policy_world_size)
    if trained_uri:
        cfg.trainer.policy.optimizer_config.lr = 1e-4
        # Four policy ranks share four examples, so each rank has one example.
        cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    # Use the same serving invariance contract as the full-Hero qualification.
    # The prior trained control run left this disabled and exposed large
    # batched-versus-single-sequence prefill differences.
    cfg.trainer.algorithm.batch_invariant = bool(trained_uri)
    cfg.trainer.policy.grug_query_bias_update_mode = "loss_free"
    cfg.trainer.policy.grug_query_bias_update_rate = 0.001
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    prompts = [[1, 17 + row, 29, 5, 11, 3] for row in range(4)]
    bias_names = [f"model.layers.{layer}.mlp.router.bias" for layer in range(model_config.num_hidden_layers)]
    names = ["model.layers.0.mlp.router.weight", *bias_names]
    if trained_uri:
        for layer in sorted({0, model_config.num_hidden_layers // 2, model_config.num_hidden_layers - 1}):
            prefix = f"model.layers.{layer}."
            names.extend(
                prefix + suffix
                for suffix in (
                    "mlp.router.weight",
                    "mlp.experts.3.gate_proj.weight",
                    "mlp.latent_down_proj.weight",
                    "mlp.latent_up_proj.weight",
                    "self_attn.q_proj.weight",
                    "self_attn.o_proj.weight",
                    "self_attn.sconv_k.weight",
                    "sconv_attn.weight",
                    "sconv_mlp.weight",
                    "shared_experts.1.up_proj.weight",
                )
            )
        names.append(
            f"model.layers.{model_config.num_hidden_layers - 1}.mlp.experts."
            f"{model_config.num_local_experts - 1}.gate_proj.weight"
        )
        names = list(dict.fromkeys(names))
    else:
        names.insert(0, "lm_head.weight")
    sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    sampling.update(temperature=0.0, max_tokens=4, ignore_eos=True, logprobs=1)
    result_uri = os.environ.get("HERO_REPLAY_RESULT_URI")

    initialize_ray(cfg)
    try:
        client = grug_engine_client(cfg, str(model_path), capture_routes=True, enable_flashinfer_autotune=False)
        policy = _init_policy(cfg, policy_world_size)
        before = rank0_validation_snapshot(policy, names)
        source_weight_diffs = _checkpoint_weight_max_diffs(model_path, before) if trained_uri else {}
        assert all(diff == 0 for diff in source_weight_diffs.values()), source_weight_diffs
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        assert_engine_weights(client, names, before, bias_names, {})

        rollout = asyncio.run(client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling)))
        assert all(len(tokens) == 4 for tokens in rollout["response_ids"])
        captured = torch.tensor(rollout["routed_experts"], dtype=torch.int32)
        assert captured.shape == (4, 4, model_config.num_hidden_layers, model_config.num_experts_per_tok)
        assert torch.all((captured >= 0) & (captured < model_config.num_local_experts))
        if result_uri:
            _put_s3_json(
                result_uri.removesuffix(".json") + "-rollout.json",
                {
                    "model": trained_uri or "random Hero schema-v2",
                    "prompts": prompts,
                    "response_ids": rollout["response_ids"],
                    "routed_experts": rollout["routed_experts"],
                },
            )
        batch = rollout_training_batch(prompts, rollout)
        native = _score(policy, batch, torch.zeros_like(captured))
        replayed = _score(policy, batch, captured)
        full_prefix_diagnostic = None
        if os.environ.get("HERO_REPLAY_DIAGNOSTIC_FULL_ROUTES") == "1":
            full_routes = torch.tensor(rollout["all_routed_experts"], dtype=torch.int32)
            assert full_routes.shape == (
                len(prompts),
                len(prompts[0]) + len(rollout["response_ids"][0]) - 1,
                model_config.num_hidden_layers,
                model_config.num_experts_per_tok,
            )
            assert torch.equal(full_routes[:, -captured.shape[1] :], captured)
            harbor_routes = []
            for prompt, response, routes in zip(prompts, rollout["response_ids"], full_routes.numpy(), strict=True):
                buffer = BytesIO()
                np.save(buffer, routes.astype(np.uint16), allow_pickle=False)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                details = [
                    {
                        "prompt_token_ids": [prompt],
                        "completion_token_ids": [response],
                        "extra": {"routed_experts": [encoded]},
                    }
                ]
                extracted = extract_full_prefix_routes_from_rollout_details(details)
                assert extracted is not None
                extracted_routes, served_tokens = extracted
                harbor_routes.append(
                    align_full_prefix_routes_to_trainer(extracted_routes, served_tokens, prompt + response)
                )
            transported = convert_prompts_responses_to_batch_tensors(
                client.tokenizer,
                prompts,
                rollout["response_ids"],
                [[0.0] * len(response) for response in rollout["response_ids"]],
                [[1] * len(response) for response in rollout["response_ids"]],
                routed_experts=harbor_routes,
                num_experts=model_config.num_local_experts,
            )
            transported_routes = transported[6]
            assert transported_routes is not None
            torch.testing.assert_close(transported_routes.int(), full_routes, rtol=0, atol=0)
            full_prefix_scores = _score(policy, batch, transported_routes)
            batch["rollout_routed_experts"] = captured
            full_prefix_diagnostic = {
                "captured_shape": list(full_routes.shape),
                "response_logprobs": full_prefix_scores.tolist(),
            }
        serving = torch.tensor(rollout["response_logprobs"], dtype=torch.float32)
        prefill = _prefill_response_logprobs(client, prompts, rollout["response_ids"], individual=False)
        single_prefill = _prefill_response_logprobs(client, prompts, rollout["response_ids"], individual=True)
        valid = batch["response_mask"].bool()
        score_diagnostic = {
            "phase": "after_score",
            "model": trained_uri or "random Hero schema-v2",
            "batch_invariant": cfg.trainer.algorithm.batch_invariant,
            "converted_split_expert_tensors": split_tensors,
            "source_weight_max_abs_diffs": source_weight_diffs,
            "native_response_logprobs": native.tolist(),
            "replay_response_logprobs": replayed.tolist(),
            "serving_response_logprobs": serving.tolist(),
            "prefill_response_logprobs": prefill.tolist(),
            "single_prefill_response_logprobs": single_prefill.tolist(),
            "prefill_vs_decode_max_abs": (prefill - serving)[valid].abs().max().item(),
            "single_vs_batch_prefill_max_abs": (single_prefill - prefill)[valid].abs().max().item(),
            "prefill_vs_megatron_native_max_abs": (prefill - native)[valid].abs().max().item(),
            "prefill_vs_megatron_replay_max_abs": (prefill - replayed)[valid].abs().max().item(),
            "native_replay_max_abs": (native - replayed)[valid].abs().max().item(),
        }
        if full_prefix_diagnostic is not None:
            full_prefix_diagnostic["vs_serving_max_abs"] = (full_prefix_scores - serving)[valid].abs().max().item()
            full_prefix_diagnostic["vs_response_only_max_abs"] = (
                (full_prefix_scores - replayed)[valid].abs().max().item()
            )
            score_diagnostic["full_prefix_diagnostic"] = full_prefix_diagnostic
        print("LIVE_HERO_REPLAY_SCORE_STATUS=" + json.dumps(score_diagnostic, sort_keys=True), flush=True)
        if result_uri:
            _put_s3_json(result_uri, score_diagnostic)
        batch["action_log_probs"] = replayed.float()
        status = _train_step(policy, batch)
        after = rank0_validation_snapshot(policy, names)
        bias_diagnostics = {
            name: {
                "finite": bool(torch.isfinite(after[name]).all()),
                "before_mean": before[name].float().mean().item(),
                "after_mean": after[name].float().mean().item(),
                "mean_delta": (after[name].float().mean() - before[name].float().mean()).item(),
                "max_change": (after[name] - before[name]).abs().max().item(),
            }
            for name in bias_names
        }
        train_diagnostic = {
            "phase": "after_train",
            "model": trained_uri or "random Hero schema-v2",
            "score_diagnostic": score_diagnostic,
            "native_logprob_max_abs": (native - serving)[valid].abs().max().item(),
            "replay_logprob_max_abs": (replayed - serving)[valid].abs().max().item(),
            "native_replay_max_abs": score_diagnostic["native_replay_max_abs"],
            "router_weight_max_change": (
                after["model.layers.0.mlp.router.weight"] - before["model.layers.0.mlp.router.weight"]
            )
            .abs()
            .max()
            .item(),
            "bias_diagnostics": bias_diagnostics,
            "status": {
                key: float(value)
                for key, value in status.items()
                if key.startswith("router_replay/")
                or key in {"raw_grad_norm", "policy_update_steps", "log_ratio_abs_max"}
            },
        }
        print("LIVE_HERO_REPLAY_TRAIN_STATUS=" + json.dumps(train_diagnostic, sort_keys=True), flush=True)
        if result_uri:
            _put_s3_json(result_uri, train_diagnostic)
        assert status["router_replay/hit_fraction"] == 1.0, status
        assert status["router_replay/executed_route_match_fraction"] == 1.0, status
        assert status["router_replay/router_grad_norm"] > 0.0, status
        assert status["router_replay/query_bias_max_change"] > 0.0, status
        assert status["log_ratio_abs_max"] < 1e-3, status
        assert not torch.equal(after["model.layers.0.mlp.router.weight"], before["model.layers.0.mlp.router.weight"])
        assert any(info["max_change"] > 0 for info in bias_diagnostics.values()), bias_diagnostics
        for name, info in bias_diagnostics.items():
            assert info["finite"], f"non-finite query bias after update: {name}"
            assert abs(info["after_mean"]) < 1e-6, f"query bias was not centered after update: {name} {info}"
        on_metrics = {
            "model": trained_uri or "random Hero schema-v2, 4 layers, 16 experts, top-8, latent MoE, ShortConv",
            "model_layers": model_config.num_hidden_layers,
            "model_experts": model_config.num_local_experts,
            "staged_checkpoint_bytes": staged_bytes,
            "converted_split_expert_tensors": split_tensors,
            "score_diagnostic": score_diagnostic,
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
    try:
        initialize_ray(cfg)
        off_policy = _init_policy(cfg, policy_world_size)
        off_weights = rank0_validation_snapshot(off_policy, names)
        off_reload_max_diffs = {
            name: (off_weights[name].float() - before[name].float()).abs().max().item() for name in names
        }
        if result_uri:
            _put_s3_json(
                result_uri, {**on_metrics, "phase": "after_off_reload", "off_reload_max_diffs": off_reload_max_diffs}
            )
        for name, max_diff in off_reload_max_diffs.items():
            assert max_diff == 0, f"flag-off reload changed {name}: max_abs={max_diff}"
        off_batch = rollout_training_batch(prompts, rollout)
        off_scores = _megatron_response_logprobs(off_policy, off_batch)
        flag_off_vs_sentinel = (off_scores - native)[valid].abs().max().item()
        assert flag_off_vs_sentinel < 1e-6, flag_off_vs_sentinel
        on_metrics["flag_off_vs_sentinel_max_abs"] = flag_off_vs_sentinel
        on_metrics["flag_off_ratio_max_deviation"] = (torch.exp(off_scores - serving)[valid] - 1).abs().max().item()
        print("LIVE_HERO_ROUTE_REPLAY=" + json.dumps(on_metrics, sort_keys=True), flush=True)
        if result_uri:
            _put_s3_json(result_uri, on_metrics)
    except Exception as exc:
        if result_uri:
            _put_s3_json(
                result_uri,
                {
                    **on_metrics,
                    "phase": "flag_off_exception",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc)[:1200],
                    "traceback": traceback.format_exc()[:6000],
                },
            )
        raise
    finally:
        ray.shutdown()


def _score(policy, batch, routes: torch.Tensor) -> torch.Tensor:
    batch["rollout_routed_experts"] = routes
    return _megatron_response_logprobs(policy, batch)


def _prefill_response_logprobs(client, prompts, responses, *, individual: bool) -> torch.Tensor:
    """Score the same tokens by vLLM prefill, separate from rollout decoding."""

    sequences = [prompt + response for prompt, response in zip(prompts, responses, strict=True)]
    sampling_params = {"temperature": 1.0, "max_tokens": 1, "prompt_logprobs": 1}
    if individual:
        rows = [
            asyncio.run(
                client.generate(InferenceEngineInput(prompt_token_ids=[sequence], sampling_params=sampling_params))
            )["prompt_logprobs"][0]
            for sequence in sequences
        ]
    else:
        rows = asyncio.run(
            client.generate(InferenceEngineInput(prompt_token_ids=sequences, sampling_params=sampling_params))
        )["prompt_logprobs"]
    assert rows is not None and len(rows) == len(prompts)
    return torch.tensor(
        [
            [row[len(prompt) + offset][token] for offset, token in enumerate(response)]
            for row, prompt, response in zip(rows, prompts, responses, strict=True)
        ],
        dtype=torch.float32,
    )


def _checkpoint_weight_max_diffs(path: Path, weights: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compare sampled loaded policy weights with the converted source export."""

    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    differences = {}
    for name, actual in weights.items():
        with safe_open(path / weight_map[name], framework="pt", device="cpu") as reader:
            expected = reader.get_tensor(name)
        if not name.endswith(".mlp.router.weight") and not name.endswith(".mlp.router.bias"):
            expected = expected.to(torch.bfloat16)
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        differences[name] = (actual.float() - expected.float()).abs().max().item()
    return differences


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
