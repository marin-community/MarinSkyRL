"""Exact rollout token transport against a local OpenAI-compatible server."""

import aiohttp
import pytest
from omegaconf import OmegaConf
from skyrl_train.config.weight_sync_pause import resolve_weight_sync_pause_policy
from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines
from aiohttp import web
from aiohttp.test_utils import TestServer

from skyrl_train.inference_engines.remote_inference_engine import RemoteInferenceEngine


class AliasingTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [2]  # Both 2 and 3 decode to the same text; retokenization loses identity.

    def decode(self, tokens, skip_special_tokens=True):
        return " X"


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", [False, True])
async def test_remote_completion_keeps_sampled_ids_and_aligned_behavior_logprobs(capture):
    requests = []

    async def generate(request):
        body = await request.json()
        requests.append(body)
        return web.json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "text": " X",
                        "token_ids": [3],
                        "finish_reason": "abort",
                        "logprobs": {
                            "token_logprobs": [-0.25],
                            "top_logprobs": [{"token_id:2": -0.75, "token_id:3": -0.25}],
                        },
                    },
                    {
                        "index": 1,
                        "text": " X",
                        "token_ids": [2],
                        "finish_reason": "stop",
                        "logprobs": {
                            "token_logprobs": [-0.5],
                            "top_logprobs": [{"token_id:3": -0.8, "token_id:2": -0.5}],
                        },
                    },
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/v1/completions", generate)
    async with TestServer(app) as server:
        engine = RemoteInferenceEngine(
            str(server.make_url("")).removeprefix("http://").rstrip("/"), "test", "vllm", AliasingTokenizer()
        )
        result = await engine.generate(
            {
                "prompt_token_ids": [[0, 1], [0, 2]],
                "sampling_params": {"temperature": 1.0, **({"logprobs": 2} if capture else {})},
            }
        )
        # An abort retry extends the original prompt with served IDs, never encoded response text.
        retry_ids = [[0, 1] + result["response_ids"][0], [0, 2] + result["response_ids"][1]]
        await engine.generate(
            {
                "prompt_token_ids": retry_ids,
                "sampling_params": {"temperature": 1.0, **({"logprobs": 2} if capture else {})},
            }
        )
    assert result["responses"] == [" X", " X"]
    assert result["response_ids"] == [[3], [2]]
    assert result["response_logprobs"] == [[-0.25], [-0.5]]
    assert result["stop_reasons"] == ["abort", "stop"]
    assert requests[0]["return_token_ids"] is True
    assert requests[0]["logprobs"] == (2 if capture else 0)
    if capture:
        assert requests[0]["return_tokens_as_token_ids"] is True
        assert result["student_topk_indices"] == [[[3, 2]], [[2, 3]]]
        assert result["behavior_topk_logprobs"] == [[[-0.25, -0.75]], [[-0.5, -0.8]]]
    else:
        assert "student_topk_indices" not in result
        assert "behavior_topk_logprobs" not in result
    assert requests[1]["prompt"] == [[0, 1, 3], [0, 2, 2]]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,clear_cache", [("abort", True), ("wait", True), ("keep", False)])
async def test_remote_pause_forwards_configured_policy_and_surfaces_rejection(mode, clear_cache):
    paused = False

    async def pause(request):
        nonlocal paused
        body = await request.json()
        assert body == {"mode": mode, "clear_cache": clear_cache}
        if mode == "wait":
            raise web.HTTPBadRequest(text="server does not support wait")
        paused = True
        return web.json_response({"status": "ok"})

    async def resume(_request):
        nonlocal paused
        paused = False
        return web.json_response({"status": "ok"})

    async def generate(_request):
        return web.json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "text": " X",
                        "token_ids": [] if paused else [3],
                        "finish_reason": "abort" if paused else "length",
                        "logprobs": {"token_logprobs": [] if paused else [-0.25]},
                    }
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/pause_generation", pause)
    app.router.add_post("/resume_generation", resume)
    app.router.add_post("/v1/completions", generate)
    generator = OmegaConf.create(
        {
            "weight_sync_pause": {"mode": mode, "clear_cache": clear_cache},
            "backend": "vllm",
            "run_engines_locally": False,
            "vllm_v1_disable_multiproc": False,
        }
    )
    async with TestServer(app) as server:
        (engine,) = create_remote_inference_engines(
            [str(server.make_url("")).removeprefix("http://").rstrip("/")],
            "test",
            "vllm",
            AliasingTokenizer(),
            weight_sync_pause_policy=resolve_weight_sync_pause_policy(generator),
        )
        request = {"prompt_token_ids": [[0, 1]], "sampling_params": {"max_tokens": 1}}
        if mode == "wait":
            with pytest.raises(aiohttp.ClientResponseError) as error:
                await engine.pause_generation()
            assert error.value.status == 400
        else:
            await engine.pause_generation()
            result = await engine.generate(request)
            assert result["stop_reasons"] == ["abort"]
            assert result["response_ids"] == result["response_logprobs"] == [[]]
            await engine.resume_generation()
        result = await engine.generate(request)
        assert result["stop_reasons"] == ["length"]
        assert result["response_ids"] == [[3]]
        assert result["response_logprobs"] == [[-0.25]]


@pytest.mark.asyncio
@pytest.mark.parametrize("omit_selected", [False, True])
async def test_remote_teacher_keeps_selected_prompt_maps_and_rejects_ignored_selection(omit_selected):
    async def generate(request):
        body = await request.json()
        assert body["prompt_logprob_token_ids"] == [[[0, 0], [2, 2]]]
        return web.json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "text": " X",
                        "token_ids": [3],
                        "finish_reason": "length",
                        "logprobs": {"token_logprobs": [-0.25]},
                        "prompt_logprobs": [
                            None,
                            {"1": {"logprob": -0.1}}
                            if omit_selected
                            else {"1": {"logprob": -0.1}, "2": {"logprob": -4.0}},
                        ],
                    }
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/v1/completions", generate)
    async with TestServer(app) as server:
        engine = RemoteInferenceEngine(
            str(server.make_url("")).removeprefix("http://").rstrip("/"),
            "test",
            "vllm",
            AliasingTokenizer(),
        )
        request = {
            "prompt_token_ids": [[0, 1]],
            "sampling_params": {"max_tokens": 1, "prompt_logprobs": 2},
            "sampling_params_per_prompt": [{"prompt_logprob_token_ids": [[0, 0], [2, 2]]}],
        }
        if omit_selected:
            with pytest.raises(ValueError, match="omitted a selected prompt token ID"):
                await engine.generate(request)
        else:
            result = await engine.generate(request)
            assert result["prompt_logprobs"] == [[None, {1: -0.1, 2: -4.0}]]
            assert result["response_ids"] == [[3]]
            assert result["response_logprobs"] == [[-0.25]]


@pytest.mark.asyncio
async def test_remote_teardown_releases_only_its_initialized_weight_group():
    active = False

    async def initialize(request):
        nonlocal active
        assert (await request.json())["group_name"] == "owned"
        active = True
        return web.json_response({"status": "initialized"})

    async def destroy(_request):
        nonlocal active
        if not active:
            raise web.HTTPConflict(text="No group is owned")
        active = False
        return web.json_response({"status": "destroyed"})

    app = web.Application()
    app.router.add_post("/init_weight_update_communicator", initialize)
    app.router.add_post("/destroy_weights_update_group", destroy)
    async with TestServer(app) as server:
        engine = RemoteInferenceEngine(
            str(server.make_url("")).removeprefix("http://").rstrip("/"),
            "test",
            "vllm",
            AliasingTokenizer(),
        )
        await engine.teardown()
        await engine.init_weight_update_communicator("127.0.0.1", 1234, 1, 2, "owned", "gloo")
        assert active
        await engine.teardown()
        assert not active
        await engine.teardown()
