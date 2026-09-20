"""Check that trained Hero prefill and rollout capture the same prefix routes."""

import asyncio
import json
import os

import numpy as np
import pytest
import ray

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import grug_engine_client
from tests.gpu.hero_replay_checkpoint import split_stacked_hero_checkpoint
from tests.gpu.test_grug_megatron import _config
from tests.gpu.test_hero_router_replay_live import _put_s3_json, _stage_trained_checkpoint


@pytest.mark.vllm
def test_trained_hero_full_prefill_routes_match_rollout(tmp_path, monkeypatch) -> None:
    trained_uri = os.environ["HERO_REPLAY_TRAINED_URI"]
    result_uri = os.environ["HERO_REPLAY_RESULT_URI"]
    assert os.environ.get("HERO_REPLAY_DIAGNOSTIC_FULL_ROUTES") == "1"
    require_hoppers(1)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    source_path = tmp_path / "hero-source"
    source_path.mkdir()
    _stage_trained_checkpoint(trained_uri, source_path)
    model_path = tmp_path / "hero"
    assert split_stacked_hero_checkpoint(source_path, model_path) > 0
    cfg = _config(str(model_path), world_size=1, pp=1, ep=1)
    cfg.trainer.algorithm.batch_invariant = True
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    prompts = [[1, 17 + row, 29, 5, 11, 3] for row in range(4)]
    sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    sampling.update(temperature=0.0, max_tokens=4, ignore_eos=True, logprobs=1)

    initialize_ray(cfg)
    try:
        client = grug_engine_client(cfg, str(model_path), capture_routes=True, enable_flashinfer_autotune=False)
        rollout = asyncio.run(client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling)))
        direct = asyncio.run(
            client.engines[0].generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling))
        )
        assert direct["response_ids"] == rollout["response_ids"]
        assert direct["routed_experts"] == rollout["routed_experts"]
        full_routes = np.asarray(direct["all_routed_experts"], dtype=np.int32)
        assert full_routes.shape == (4, 9, 16, 8)

        prefill_routes = []
        prefill_scores = []
        for prompt, response in zip(prompts, rollout["response_ids"], strict=True):
            sequence = prompt + response
            single = asyncio.run(
                client.engines[0].generate(
                    InferenceEngineInput(
                        prompt_token_ids=[sequence],
                        sampling_params={"temperature": 1.0, "max_tokens": 1, "prompt_logprobs": 1},
                    )
                )
            )
            routes = np.asarray(single["all_routed_experts"][0], dtype=np.int32)
            assert routes.shape[0] >= len(sequence) - 1
            prefill_routes.append(routes[: len(sequence) - 1])
            token_scores = single["prompt_logprobs"][0]
            prefill_scores.append([token_scores[len(prompt) + offset][token] for offset, token in enumerate(response)])
        prefill_routes_array = np.stack(prefill_routes)
        route_mismatches = (prefill_routes_array != full_routes).any(axis=-1)
        serving_scores = np.asarray(rollout["response_logprobs"], dtype=np.float32)
        score_max_abs = float(np.max(np.abs(np.asarray(prefill_scores, dtype=np.float32) - serving_scores)))
        report = {
            "model": trained_uri,
            "source_commit": os.environ.get("HERO_REPLAY_SOURCE_COMMIT"),
            "prompts": prompts,
            "response_ids": rollout["response_ids"],
            "captured_shape": list(full_routes.shape),
            "route_mismatch_count": int(route_mismatches.sum()),
            "route_mismatch_by_row": route_mismatches.sum(axis=(1, 2)).tolist(),
            "prefill_vs_decode_logprob_max_abs": score_max_abs,
            "rollout_full_routes": full_routes.tolist(),
            "prefill_full_routes": prefill_routes_array.tolist(),
        }
        _put_s3_json(result_uri, report)
        summary = {key: value for key, value in report.items() if not key.endswith("_full_routes")}
        print("HERO_PREFILL_ROUTE_PARITY=" + json.dumps(summary, sort_keys=True), flush=True)
        assert report["route_mismatch_count"] == 0, report["route_mismatch_by_row"]
        assert score_max_abs == 0, score_max_abs
    finally:
        ray.shutdown()
