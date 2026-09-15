from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
import torch
from aiohttp import web
from omegaconf import OmegaConf
from rigging.secrets import SecretResolutionError

import skyrl_train.teacher_runtime as runtime_module
from skyrl_train.teacher_runtime import (
    prepare_async_distillation_runtime,
    prepare_distillation_runtime,
    start_async_distillation_runtime,
    start_sync_distillation_runtime,
)
from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import TrajectoryID
from tests.fixtures.opd_http_teacher import application


@dataclass
class _Tokenizer:
    vocabulary: dict[str, int]

    def get_vocab(self):
        return self.vocabulary


class _Engine:
    def __init__(self, *, model_max_len: int | None = 32) -> None:
        self.model_max_len = model_max_len
        self.teardown_count = 0
        self.requests = []

    def get_model_max_len(self):
        return self.model_max_len

    async def generate(self, request):
        self.requests.append(request)
        prompt_logprobs = []
        for sequence in request["prompt_token_ids"]:
            prompt_logprobs.append([None, *({token_id: -0.25} for token_id in sequence[1:])])
        return {
            "responses": [""] * len(prompt_logprobs),
            "response_ids": [[]] * len(prompt_logprobs),
            "stop_reasons": ["length"] * len(prompt_logprobs),
            "response_logprobs": None,
            "prompt_logprobs": prompt_logprobs,
        }

    async def teardown(self):
        self.teardown_count += 1


def _config():
    return OmegaConf.create(
        {
            "trainer": {
                "algorithm": {
                    "distillation": {
                        "objective": "sampled_reverse_kl",
                        "routing_plan": "opd",
                        "coefficient": 0.5,
                        "reward_mode": "add",
                    }
                },
                "disable_fast_tokenizer": False,
                "seed": 7,
                "distributed": {"placement_group_timeout_seconds": 30},
                "fully_async": {"teacher_scoring": {"max_queued_per_teacher": 1, "workers_per_teacher": 1}},
            },
            "generator": {
                "model_dtype": "bfloat16",
                "vllm_v1_disable_multiproc": True,
                "enable_prefix_caching": False,
                "enforce_eager": True,
                "engine_init_timeout_seconds": 30,
                "gpu_memory_utilization": 0.5,
                "async_engine": False,
                "max_num_batched_tokens": 128,
                "max_num_seqs": 4,
                "engine_init_kwargs": {},
            },
            "teachers": {
                "primary": {
                    "source": "local_inference",
                    "placement": "pinned",
                    "model": {"path": "Qwen/teacher", "revision": "teacher-revision"},
                    "backend": "vllm",
                    "evidence": "chosen_token",
                    "resources": {
                        "num_nodes": 1,
                        "gpus_per_node": 1,
                        "tensor_parallel_size": 1,
                        "colocation_group": "teacher",
                    },
                }
            },
            "teacher_routing": {
                "opd": {
                    "revision": "route-revision",
                    "routes": {"default": {"teacher": "primary", "weight": 1.0}},
                }
            },
        }
    )


def _two_teacher_config(*, placement: str):
    cfg = _config()
    cfg.trainer.algorithm.distillation.residency = {
        "max_resident": 1,
        "minimum_residency_seconds": 0,
    }
    cfg.teachers.primary.placement = placement
    cfg.teachers.secondary = {
        "source": "local_inference",
        "placement": placement,
        "model": {"path": "Qwen/teacher-secondary", "revision": "teacher-secondary-revision"},
        "backend": "vllm",
        "evidence": "chosen_token",
        "resources": {
            "num_nodes": 1,
            "gpus_per_node": 1,
            "tensor_parallel_size": 1,
            "colocation_group": "teacher" if placement == "rotating" else "teacher-secondary",
        },
    }
    cfg.teacher_routing.opd.routes = {
        "primary": {"teacher": "primary", "weight": 1.0},
        "secondary": {"teacher": "secondary", "weight": 1.0},
    }
    return cfg


def _add_external_teacher(cfg, teacher_id: str, base_url: str, tokenizer: _Tokenizer):
    cfg.teachers[teacher_id] = {
        "source": "openai_compatible",
        "placement": "external",
        "model": {"path": "Qwen/remote-teacher", "revision": "remote-revision"},
        "endpoints": [
            {
                "url": base_url,
                "auth": "env:REMOTE_TEACHER_API_KEY",
                "max_concurrency": 2,
            }
        ],
        "tokenizer_fingerprint": tokenizer_vocabulary_fingerprint(tokenizer),
        "max_sequence_length": 32,
        "request_timeout_seconds": 5,
        "evidence": "chosen_token",
    }
    return cfg


def _external_teacher_config(base_url: str, tokenizer: _Tokenizer):
    return _add_external_teacher(_config(), "primary", base_url, tokenizer)


def _mixed_teacher_config(base_url: str, tokenizer: _Tokenizer):
    return _add_external_teacher(_two_teacher_config(placement="pinned"), "secondary", base_url, tokenizer)


@asynccontextmanager
async def _remote_teacher_server(port: int, requests: list[dict]):
    app = application(-0.75, requests=requests, bearer_token="test-api-key")
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        await runner.cleanup()


async def _score_two_routes(runtime):
    _, scored = await runtime.score_while_model_forwarding(
        {
            "trajectory_ids": [TrajectoryID("math", 0), TrajectoryID("code", 0)],
            "teacher_route_keys": ["primary", "secondary"],
            "prompt_token_ids": [[0, 1], [0, 2]],
            "response_ids": [[2], [1]],
        },
        lambda: TrainingInputBatch({"policy": torch.tensor([1.0])}),
    )
    return scored


@pytest.mark.asyncio
async def test_local_teacher_runtime_rejects_tokenizer_mismatch_before_engine_allocation(monkeypatch):
    allocation_attempted = False

    def create_engines(**_kwargs):
        nonlocal allocation_attempted
        allocation_attempted = True
        return []

    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: _Tokenizer({"b": 0}))
    monkeypatch.setattr(runtime_module, "create_ray_wrapped_inference_engines", create_engines)

    with pytest.raises(ValueError, match="tokenizer vocabulary does not match"):
        prepare_distillation_runtime(_config(), _Tokenizer({"a": 0}))

    assert not allocation_attempted


def test_local_teacher_runtime_accepts_multiple_routes_to_one_pinned_teacher(monkeypatch):
    tokenizer = _Tokenizer({"a": 0})
    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    cfg = _config()
    cfg.teacher_routing.opd.routes.math = {"teacher": "primary", "weight": 0.75}

    prepared = prepare_distillation_runtime(cfg, tokenizer)

    assert prepared is not None
    assert [(route.key, route.teacher_id) for route in prepared.plan.routing.routes] == [
        ("default", "primary"),
        ("math", "primary"),
    ]


def test_fully_async_teacher_queue_limits_fail_before_teacher_initialization(monkeypatch):
    tokenizer_initialized = False

    def create_teacher_tokenizer(*_args, **_kwargs):
        nonlocal tokenizer_initialized
        tokenizer_initialized = True
        return _Tokenizer({"a": 0})

    monkeypatch.setattr(runtime_module, "create_tokenizer", create_teacher_tokenizer)
    cfg = _config()
    cfg.trainer.fully_async.teacher_scoring.max_queued_per_teacher = 0

    with pytest.raises(ValueError, match="queue and worker limits must be positive"):
        prepare_async_distillation_runtime(cfg, _Tokenizer({"a": 0}))

    assert not tokenizer_initialized


@pytest.mark.asyncio
async def test_local_teacher_runtime_scores_exact_rollout_tokens_and_owns_engine(monkeypatch):
    engine = _Engine()
    engine_kwargs = {}
    tokenizer = _Tokenizer({"a": 0, "b": 1, "c": 2})

    def create_engine(**kwargs):
        engine_kwargs.update(kwargs)
        return [engine]

    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(runtime_module, "create_ray_wrapped_inference_engines", create_engine)
    cfg = _config()
    cfg.teachers.primary.resources.max_num_batched_tokens = 64
    prepared = prepare_distillation_runtime(cfg, tokenizer)
    runtime = await start_sync_distillation_runtime(cfg, prepared)
    assert runtime is not None

    forwarded, scored = await runtime.score_while_model_forwarding(
        {
            "trajectory_ids": [TrajectoryID("math", 0)],
            "prompt_token_ids": [[0, 1]],
            "response_ids": [[2, 1]],
        },
        lambda: TrainingInputBatch({"policy": torch.tensor([1.0])}),
    )
    await runtime.close()

    torch.testing.assert_close(forwarded["policy"], torch.tensor([1.0]))
    torch.testing.assert_close(scored.distillation.teacher_action_log_probs, torch.tensor([[-0.25, -0.25]]))
    torch.testing.assert_close(scored.distillation.loss_weights, torch.tensor([[0.5, 0.5]]))
    assert engine_kwargs["max_num_batched_tokens"] == 64
    assert engine.teardown_count == 1


@pytest.mark.asyncio
async def test_local_teacher_runtime_feeds_fully_async_admitted_groups(monkeypatch):
    engine = _Engine()
    tokenizer = _Tokenizer({"a": 0, "b": 1, "c": 2})
    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(runtime_module, "create_ray_wrapped_inference_engines", lambda **_kwargs: [engine])
    cfg = _config()
    prepared = prepare_async_distillation_runtime(cfg, tokenizer)
    runtime = await start_async_distillation_runtime(cfg, prepared)
    assert runtime is not None
    await runtime.start()

    ticket = await runtime.submit_before_batch_assembly(
        {
            "trajectory_ids": [TrajectoryID("math", 0)],
            "prompt_token_ids": [[0, 1]],
            "response_ids": [[2, 1]],
        }
    )
    scored = await ticket.result()
    training_input = TrainingInputBatch({"response_mask": torch.ones((1, 2), dtype=torch.bool)})
    training_input.metadata = {"pad_size": 0}
    runtime.attach_to_training_input(training_input, (scored,))
    await runtime.close()

    torch.testing.assert_close(
        training_input["teacher_action_log_probs"],
        torch.tensor([[-0.25, -0.25]]),
    )
    assert engine.teardown_count == 1


@pytest.mark.asyncio
async def test_teacher_runtime_composes_remote_and_local_routes(monkeypatch, unused_tcp_port):
    engine = _Engine()
    tokenizer = _Tokenizer({"a": 0, "b": 1, "c": 2})
    remote_requests = []
    monkeypatch.setenv("REMOTE_TEACHER_API_KEY", "test-api-key")
    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(runtime_module, "create_ray_wrapped_inference_engines", lambda **_kwargs: [engine])

    async with _remote_teacher_server(unused_tcp_port, remote_requests) as base_url:
        cfg = _mixed_teacher_config(base_url, tokenizer)
        prepared = prepare_distillation_runtime(cfg, tokenizer)
        runtime = await start_sync_distillation_runtime(cfg, prepared)
        assert runtime is not None
        scored = await _score_two_routes(runtime)
        await runtime.close()

    torch.testing.assert_close(
        scored.distillation.teacher_action_log_probs,
        torch.tensor([[-0.25], [-0.75]]),
    )
    assert tuple(route.teacher_id for route in scored.routes) == ("primary", "secondary")
    assert remote_requests[0]["prompt"] == [[0, 2, 1]]
    assert engine.teardown_count == 1


def test_remote_teacher_runtime_rejects_tokenizer_mismatch_before_startup():
    tokenizer = _Tokenizer({"a": 0, "b": 1, "c": 2})
    cfg = _external_teacher_config("https://teacher.example/v1", tokenizer)
    cfg.teachers.primary.tokenizer_fingerprint = f"sha256:{'f' * 64}"

    with pytest.raises(ValueError, match="tokenizer fingerprint does not match"):
        prepare_distillation_runtime(cfg, tokenizer)


@pytest.mark.asyncio
async def test_remote_teacher_runtime_rejects_missing_auth_secret(monkeypatch):
    tokenizer = _Tokenizer({"a": 0, "b": 1, "c": 2})
    cfg = _external_teacher_config("https://teacher.example/v1", tokenizer)
    monkeypatch.delenv("REMOTE_TEACHER_API_KEY", raising=False)
    prepared = prepare_distillation_runtime(cfg, tokenizer)

    with pytest.raises(SecretResolutionError, match="no secret source produced a value"):
        await start_sync_distillation_runtime(cfg, prepared)


@pytest.mark.asyncio
async def test_remote_teacher_runtime_feeds_fully_async_admitted_groups(monkeypatch, unused_tcp_port):
    tokenizer = _Tokenizer({"a": 0, "b": 1, "c": 2})
    remote_requests = []
    monkeypatch.setenv("REMOTE_TEACHER_API_KEY", "test-api-key")

    async with _remote_teacher_server(unused_tcp_port, remote_requests) as base_url:
        cfg = _external_teacher_config(base_url, tokenizer)
        prepared = prepare_async_distillation_runtime(cfg, tokenizer)
        runtime = await start_async_distillation_runtime(cfg, prepared)
        assert runtime is not None
        await runtime.start()
        ticket = await runtime.submit_before_batch_assembly(
            {
                "trajectory_ids": [TrajectoryID("math", 0)],
                "prompt_token_ids": [[0, 1]],
                "response_ids": [[2, 1]],
            }
        )
        scored = await ticket.result()
        training_input = TrainingInputBatch({"response_mask": torch.ones((1, 2), dtype=torch.bool)})
        training_input.metadata = {"pad_size": 0}
        runtime.attach_to_training_input(training_input, (scored,))
        await runtime.close()

    torch.testing.assert_close(training_input["teacher_action_log_probs"], torch.tensor([[-0.75, -0.75]]))
    assert remote_requests[0]["prompt"] == [[0, 1, 2, 1]]


@pytest.mark.asyncio
async def test_local_teacher_runtime_cleans_engine_when_oracle_startup_fails(monkeypatch):
    engine = _Engine(model_max_len=None)
    tokenizer = _Tokenizer({"a": 0})
    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(runtime_module, "create_ray_wrapped_inference_engines", lambda **_kwargs: [engine])
    cfg = _config()
    prepared = prepare_distillation_runtime(cfg, tokenizer)

    with pytest.raises(ValueError, match="maximum model length"):
        await start_sync_distillation_runtime(cfg, prepared)

    assert engine.teardown_count == 1


@pytest.mark.asyncio
async def test_local_teacher_runtime_eagerly_owns_multiple_pinned_teachers(monkeypatch):
    tokenizer = _Tokenizer({"a": 0, "b": 1, "c": 2})
    pending_engines = [_Engine(), _Engine()]
    started_engines = []

    def create_engines(**_kwargs):
        engine = pending_engines.pop(0)
        started_engines.append(engine)
        return [engine]

    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(runtime_module, "create_ray_wrapped_inference_engines", create_engines)
    cfg = _two_teacher_config(placement="pinned")
    prepared = prepare_distillation_runtime(cfg, tokenizer)
    runtime = await start_sync_distillation_runtime(cfg, prepared)
    assert runtime is not None
    assert len(started_engines) == 2

    scored = await _score_two_routes(runtime)
    await runtime.close()

    assert tuple(route.teacher_id for route in scored.routes) == ("primary", "secondary")
    assert [len(engine.requests) for engine in started_engines] == [1, 1]
    assert [engine.teardown_count for engine in started_engines] == [1, 1]


@pytest.mark.asyncio
async def test_local_teacher_runtime_cleans_started_pinned_teacher_when_later_startup_fails(monkeypatch):
    tokenizer = _Tokenizer({"a": 0})
    engines = [_Engine(), _Engine(model_max_len=None)]
    pending_engines = list(engines)
    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(
        runtime_module, "create_ray_wrapped_inference_engines", lambda **_kwargs: [pending_engines.pop(0)]
    )
    cfg = _two_teacher_config(placement="pinned")
    prepared = prepare_distillation_runtime(cfg, tokenizer)

    with pytest.raises(ValueError, match="maximum model length"):
        await start_sync_distillation_runtime(cfg, prepared)

    assert pending_engines == []
    assert [engine.teardown_count for engine in engines] == [1, 1]


@pytest.mark.asyncio
async def test_local_teacher_runtime_rotates_teachers_on_one_residency_slot(monkeypatch):
    tokenizer = _Tokenizer({"a": 0, "b": 1, "c": 2})
    started_engines = []

    def create_engines(**_kwargs):
        engine = _Engine()
        started_engines.append(engine)
        return [engine]

    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(runtime_module, "create_ray_wrapped_inference_engines", create_engines)
    cfg = _two_teacher_config(placement="rotating")
    prepared = prepare_distillation_runtime(cfg, tokenizer)
    runtime = await start_sync_distillation_runtime(cfg, prepared)
    assert runtime is not None
    assert started_engines == []

    scored = await _score_two_routes(runtime)
    await runtime.close()

    assert tuple(route.teacher_id for route in scored.routes) == ("primary", "secondary")
    assert len(started_engines) == 2
    assert [len(engine.requests) for engine in started_engines] == [1, 1]
    assert [engine.teardown_count for engine in started_engines] == [1, 1]


@pytest.mark.parametrize(
    ("placements", "groups", "message"),
    [
        (("pinned", "pinned"), ("teacher", "teacher"), "pinned local teachers must use distinct"),
        (("rotating", "rotating"), ("teacher", "other"), "share one identical resource footprint"),
        (("pinned", "rotating"), ("teacher", "teacher"), "cannot share colocation group"),
    ],
)
def test_local_teacher_runtime_rejects_unsafe_multi_teacher_resource_layouts(monkeypatch, placements, groups, message):
    tokenizer = _Tokenizer({"a": 0})
    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    cfg = _two_teacher_config(placement="rotating")
    cfg.teachers.primary.placement, cfg.teachers.secondary.placement = placements
    cfg.teachers.primary.resources.colocation_group, cfg.teachers.secondary.resources.colocation_group = groups

    with pytest.raises(ValueError, match=message):
        prepare_distillation_runtime(cfg, tokenizer)


def test_local_teacher_runtime_rejects_unplanned_additional_residency_slots(monkeypatch):
    tokenizer = _Tokenizer({"a": 0})
    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    cfg = _two_teacher_config(placement="rotating")
    cfg.trainer.algorithm.distillation.residency.max_resident = 2

    with pytest.raises(ValueError, match="exactly one rotating residency slot"):
        prepare_distillation_runtime(cfg, tokenizer)
