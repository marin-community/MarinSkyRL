from __future__ import annotations

import math
import asyncio
from contextlib import asynccontextmanager

import pytest
import torch
from aiohttp import web

from marinskyrl.distillation import (
    OpenAICompatibleTeacherSpec,
    TeacherEndpointSpec,
    TeacherEvidenceKind,
    TeacherModelSpec,
    TeacherPlacement,
    TeacherSource,
)
from skyrl_train.distillation import ChosenTokenTeacherEvidence, TeacherScoreRequest, TopKTeacherEvidence
from skyrl_train.inference_engines.openai_teacher_oracle import OpenAICompatibleTeacherOracle
from skyrl_train.teacher_oracle import TeacherEndpointUnavailable
from tests.fixtures.opd_http_teacher import application, completion_choice

_FINGERPRINT = f"sha256:{'a' * 64}"


def _teacher(evidence: TeacherEvidenceKind) -> OpenAICompatibleTeacherSpec:
    return OpenAICompatibleTeacherSpec(
        id="teacher",
        source=TeacherSource.OPENAI_COMPATIBLE,
        placement=TeacherPlacement.EXTERNAL,
        model=TeacherModelSpec(path="Qwen/teacher", revision="teacher-r7"),
        evidence=evidence,
        endpoints=(),
        tokenizer_fingerprint=_FINGERPRINT,
        max_sequence_length=32,
        request_timeout_seconds=5,
        top_k=2 if evidence is TeacherEvidenceKind.TOPK_DISTRIBUTION else None,
    )


def _request(evidence: TeacherEvidenceKind) -> TeacherScoreRequest:
    return TeacherScoreRequest(
        trajectory_ids=("row-0",),
        route_ids=("math",),
        teacher_id="teacher",
        tokenizer_fingerprint=_FINGERPRINT,
        plan_version="routes-r2",
        prompt_token_ids=torch.tensor([[0, 1]]),
        prompt_mask=torch.tensor([[True, True]]),
        response_token_ids=torch.tensor([[2, 1, 0]]),
        response_mask=torch.tensor([[True, True, False]]),
        evidence=evidence,
        top_k=2 if evidence is TeacherEvidenceKind.TOPK_DISTRIBUTION else None,
    )


@asynccontextmanager
async def _server(app: web.Application, port: int):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        await runner.cleanup()


def _choice(prompt: list[int]) -> dict:
    choice = completion_choice(0, prompt, math.log(0.5))
    chosen_scores = [None, math.log(0.8), math.log(0.6), math.log(0.5)]
    top_scores = [
        None,
        {"token_id:1": math.log(0.8), "token_id:0": math.log(0.1)},
        {"token_id:2": math.log(0.6), "token_id:1": math.log(0.25)},
        {"token_id:1": math.log(0.5), "token_id:2": math.log(0.3)},
    ]
    choice["logprobs"]["token_logprobs"] = chosen_scores
    choice["logprobs"]["top_logprobs"] = top_scores
    return choice


def test_openai_teacher_oracle_binds_http_session_to_scoring_loop(unused_tcp_port):
    async def construct():
        return OpenAICompatibleTeacherOracle(
            teacher=_teacher(TeacherEvidenceKind.CHOSEN_TOKEN),
            endpoint=TeacherEndpointSpec(
                url=f"http://127.0.0.1:{unused_tcp_port}/v1",
                auth=None,
                max_concurrency=1,
            ),
            api_key=None,
        )

    oracle = asyncio.run(construct())

    async def score_on_new_loop():
        app = application(-0.25, choice_factory=lambda _index, sequence: _choice(sequence))
        async with _server(app, unused_tcp_port):
            evidence = await oracle.score(_request(TeacherEvidenceKind.CHOSEN_TOKEN))
            await oracle.close()
            return evidence

    evidence = asyncio.run(score_on_new_loop())

    assert isinstance(evidence, ChosenTokenTeacherEvidence)


@pytest.mark.asyncio
async def test_openai_teacher_oracle_scores_exact_tokens_with_bearer_auth(unused_tcp_port):
    requests = []

    app = application(
        -0.25,
        requests=requests,
        bearer_token="private-key",
        choice_factory=lambda _index, sequence: _choice(sequence),
    )
    async with _server(app, unused_tcp_port) as base_url:
        endpoint = TeacherEndpointSpec(url=base_url, auth=None, max_concurrency=3)
        oracle = OpenAICompatibleTeacherOracle(
            teacher=_teacher(TeacherEvidenceKind.CHOSEN_TOKEN),
            endpoint=endpoint,
            api_key="private-key",
        )
        evidence = await oracle.score(_request(TeacherEvidenceKind.CHOSEN_TOKEN))
        await oracle.close()

    assert isinstance(evidence, ChosenTokenTeacherEvidence)
    torch.testing.assert_close(
        evidence.chosen_logprobs,
        torch.tensor([[math.log(0.6), math.log(0.5), torch.nan]]),
        equal_nan=True,
    )
    assert requests == [
        {
            "model": "Qwen/teacher",
            "prompt": [[0, 1, 2, 1]],
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": 0,
            "return_tokens_as_token_ids": True,
        }
    ]


@pytest.mark.asyncio
async def test_openai_teacher_oracle_normalizes_topk_evidence(unused_tcp_port):
    app = application(-0.25, choice_factory=lambda _index, sequence: _choice(sequence))
    async with _server(app, unused_tcp_port) as base_url:
        oracle = OpenAICompatibleTeacherOracle(
            teacher=_teacher(TeacherEvidenceKind.TOPK_DISTRIBUTION),
            endpoint=TeacherEndpointSpec(url=base_url, auth=None, max_concurrency=2),
            api_key=None,
        )
        evidence = await oracle.score(_request(TeacherEvidenceKind.TOPK_DISTRIBUTION))
        await oracle.close()

    assert isinstance(evidence, TopKTeacherEvidence)
    torch.testing.assert_close(evidence.topk_indices, torch.tensor([[[2, 1], [1, 2], [-1, -1]]]))
    torch.testing.assert_close(evidence.retained_mass[0, :2], torch.tensor([0.85, 0.8]))


@pytest.mark.asyncio
async def test_openai_teacher_oracle_rejects_changed_token_identity(unused_tcp_port):
    app = application(-0.25, choice_factory=lambda _index, _sequence: _choice([0, 1, 2, 0]))
    async with _server(app, unused_tcp_port) as base_url:
        oracle = OpenAICompatibleTeacherOracle(
            teacher=_teacher(TeacherEvidenceKind.CHOSEN_TOKEN),
            endpoint=TeacherEndpointSpec(url=base_url, auth=None, max_concurrency=1),
            api_key=None,
        )
        with pytest.raises(ValueError, match="changed token identity"):
            await oracle.score(_request(TeacherEvidenceKind.CHOSEN_TOKEN))
        await oracle.close()


@pytest.mark.asyncio
async def test_openai_teacher_oracle_marks_http_failures_retryable(unused_tcp_port):
    async def handle(_request: web.Request):
        return web.json_response({"error": "capacity"}, status=503)

    app = web.Application()
    app.router.add_post("/v1/completions", handle)
    async with _server(app, unused_tcp_port) as base_url:
        oracle = OpenAICompatibleTeacherOracle(
            teacher=_teacher(TeacherEvidenceKind.CHOSEN_TOKEN),
            endpoint=TeacherEndpointSpec(url=base_url, auth=None, max_concurrency=1),
            api_key=None,
        )
        with pytest.raises(TeacherEndpointUnavailable, match="HTTP 503"):
            await oracle.score(_request(TeacherEvidenceKind.CHOSEN_TOKEN))
        await oracle.close()
