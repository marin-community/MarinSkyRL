"""Compare trained Hero vLLM and Megatron layer boundaries on identical tokens and routes."""

import asyncio
from io import BytesIO
import json
import os

import numpy as np
import pytest
import ray
import torch

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.models.grug_moe import GrugMoeConfig
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import grug_engine_client, rollout_training_batch
from tests.gpu.hero_replay_checkpoint import split_stacked_hero_checkpoint
from tests.gpu.test_grug_megatron import _config, _init_policy
from tests.gpu.test_hero_router_replay_live import _s3_client, _s3_target, _score, _stage_trained_checkpoint


@pytest.mark.vllm
def test_trained_hero_full_prefix_layer_trace(tmp_path, monkeypatch) -> None:
    trained_uri = os.environ["HERO_REPLAY_TRAINED_URI"]
    result_uri = os.environ["HERO_REPLAY_RESULT_URI"]
    assert os.environ.get("HERO_REPLAY_DIAGNOSTIC_FULL_ROUTES") == "1"
    require_hoppers(5)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    source_path = tmp_path / "hero-source"
    source_path.mkdir()
    _stage_trained_checkpoint(trained_uri, source_path)
    model_path = tmp_path / "hero"
    assert split_stacked_hero_checkpoint(source_path, model_path) > 0
    model_config = GrugMoeConfig.from_pretrained(model_path)
    assert model_config.num_hidden_layers == 16

    cfg = _config(str(model_path), world_size=4, pp=1, ep=4)
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    cfg.trainer.algorithm.batch_invariant = True
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    prompts = [[1, 17 + row, 29, 5, 11, 3] for row in range(4)]
    sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    sampling.update(temperature=0.0, max_tokens=4, ignore_eos=True, logprobs=1)
    positions = [0, 1, 5, 6, 7, 8]
    row_index = 3

    initialize_ray(cfg)
    try:
        client = grug_engine_client(cfg, str(model_path), capture_routes=True, enable_flashinfer_autotune=False)
        policy = _init_policy(cfg, 4)
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

        rollout = asyncio.run(client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling)))
        direct = asyncio.run(
            client.engines[0].generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling))
        )
        assert direct["response_ids"] == rollout["response_ids"]
        assert direct["routed_experts"] == rollout["routed_experts"]
        full_routes = torch.tensor(direct["all_routed_experts"], dtype=torch.int32)
        assert full_routes.shape == (4, 9, model_config.num_hidden_layers, model_config.num_experts_per_tok)

        sequence = prompts[row_index] + rollout["response_ids"][row_index]
        vllm_actor = client.engines[0].inference_engine_actor
        vllm_start = ray.get(vllm_actor.begin_hero_layer_trace.remote(positions))
        if isinstance(vllm_start, dict):
            vllm_start = [vllm_start]
        assert vllm_start[0]["layers"] == model_config.num_hidden_layers
        try:
            asyncio.run(
                client.engines[0].generate(
                    InferenceEngineInput(
                        prompt_token_ids=[sequence],
                        sampling_params={"temperature": 1.0, "max_tokens": 1, "prompt_logprobs": 1},
                    )
                )
            )
        finally:
            vllm_rank_traces = ray.get(vllm_actor.finish_hero_layer_trace.remote())
        if isinstance(vllm_rank_traces, dict):
            vllm_rank_traces = [vllm_rank_traces]
        assert len(vllm_rank_traces) == 1
        vllm_trace = vllm_rank_traces[0]

        mega_start = ray.get(policy.async_run_ray_method("pass_through", "begin_grug_layer_trace", len(sequence)))
        assert len(mega_start) == 4
        assert all(item["layers"] == model_config.num_hidden_layers for item in mega_start)
        batch = rollout_training_batch(prompts, rollout)
        try:
            scores = _score(policy, batch, full_routes)
        finally:
            mega_rank_traces = ray.get(policy.async_run_ray_method("pass_through", "finish_grug_layer_trace"))
        # EP4 distributes the four sequences across ranks. Identify the row
        # from its second-token embedding, rather than assuming a rank order.
        embed_key = "layer_0_model_input"
        serving_embed = vllm_trace[embed_key][positions.index(1)].numpy()
        embedding_rms_by_rank = {}
        for item in mega_rank_traces:
            calls = item["traces"][embed_key]
            assert len(calls) == 1 and calls[0].shape[:2] == (len(sequence), 1)
            delta = calls[0][1, 0].numpy() - serving_embed
            embedding_rms_by_rank[item["rank"]] = float(np.sqrt(np.mean(np.square(delta))))
        closest = sorted(embedding_rms_by_rank, key=embedding_rms_by_rank.get)
        assert embedding_rms_by_rank[closest[0]] < embedding_rms_by_rank[closest[1]] / 2
        selected_rank = closest[0]
        mega_trace = next(item for item in mega_rank_traces if item["rank"] == selected_rank)["traces"]

        trace_arrays = {"positions": np.asarray(positions), "token_ids": np.asarray(sequence)}
        metrics = {}
        for layer in range(model_config.num_hidden_layers):
            metrics[str(layer)] = {}
            sites = ["model_input", "after_attn", "mlp_input", "after_block"]
            if layer == 0:
                sites.extend(
                    ("routed_latent", "routed_expanded", "shared_0", "shared_1", "before_sconv_mlp", "after_sconv_mlp")
                )
            for site in sites:
                key = f"layer_{layer}_{site}"
                serving = vllm_trace[key].numpy()
                calls = mega_trace[key]
                assert len(calls) == 1, (key, len(calls))
                assert calls[0].shape[:2] == (len(sequence), 1), (
                    key,
                    [call.shape for call in calls],
                )
                trainer = calls[0][positions, 0].numpy()
                width = model_config.latent_dim if site == "routed_latent" else model_config.hidden_size
                assert serving.shape == trainer.shape == (len(positions), width)
                assert np.isfinite(serving).all() and np.isfinite(trainer).all(), key
                delta = trainer - serving
                metrics[str(layer)][site] = {
                    "rms_by_position": np.sqrt(np.mean(np.square(delta), axis=-1)).tolist(),
                    "max_abs_by_position": np.max(np.abs(delta), axis=-1).tolist(),
                }
                trace_arrays[f"vllm_{key}"] = serving
                trace_arrays[f"megatron_{key}"] = trainer

        trace_uri = result_uri.removesuffix(".json") + "-arrays.npz"
        payload = BytesIO()
        np.savez_compressed(payload, **trace_arrays)
        bucket, key = _s3_target(trace_uri)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=payload.getvalue())
        rollout_scores = torch.tensor(rollout["response_logprobs"], dtype=torch.float32)
        max_response_gap = (scores - rollout_scores).abs().max().item()
        # The same full-prefix control measured 0.05281 before observation hooks.
        # A larger shift would make the boundary trace an altered workload.
        assert max_response_gap < 0.1, max_response_gap
        report = {
            "model": trained_uri,
            "layout": "vLLM TP1/EP1; Megatron TP1/PP1/EP4/CP1",
            "source_commit": os.environ.get("HERO_REPLAY_SOURCE_COMMIT"),
            "row_index": row_index,
            "selected_megatron_rank": selected_rank,
            "embedding_rms_by_rank": embedding_rms_by_rank,
            "sequence": sequence,
            "positions": positions,
            "captured_shape": list(full_routes.shape),
            "response_scores": scores.tolist(),
            "serving_response_scores": rollout_scores.tolist(),
            "max_response_logprob_gap": max_response_gap,
            "raw_arrays_uri": trace_uri,
            "layers": metrics,
        }
        print("HERO_LAYER_TRACE_STATUS=" + json.dumps(report, sort_keys=True), flush=True)
        result_bucket, result_key = _s3_target(result_uri)
        _s3_client().put_object(Bucket=result_bucket, Key=result_key, Body=json.dumps(report).encode())
    finally:
        ray.shutdown()
