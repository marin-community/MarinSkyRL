from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.distillation import ChosenTokenTeacherEvidence, TeacherScoreRequest, TopKTeacherEvidence
from skyrl_train.inference_engines.vllm_teacher_oracle import VLLMTeacherOracle, tokenizer_vocabulary_fingerprint


@dataclass
class _Tokenizer:
    vocabulary: dict[str, int]

    def get_vocab(self):
        return self.vocabulary


class _Engine:
    def __init__(self, prompt_logprobs):
        self.prompt_logprobs = prompt_logprobs
        self.requests = []
        self.teardown_calls = 0

    def get_model_max_len(self):
        return 32

    async def generate(self, request):
        self.requests.append(request)
        return {
            "responses": [""],
            "response_ids": [[]],
            "stop_reasons": ["length"],
            "response_logprobs": None,
            "prompt_logprobs": self.prompt_logprobs,
        }

    async def teardown(self):
        self.teardown_calls += 1


def _request(evidence: TeacherEvidenceKind, *, top_k: int | None = None) -> TeacherScoreRequest:
    return TeacherScoreRequest(
        trajectory_ids=("row-0",),
        route_ids=("math",),
        teacher_id="teacher",
        tokenizer_fingerprint=tokenizer_vocabulary_fingerprint(_Tokenizer({"a": 0, "b": 1, "c": 2})),
        plan_version="routing-v1",
        prompt_token_ids=torch.tensor([[0, 1]]),
        prompt_mask=torch.tensor([[True, True]]),
        response_token_ids=torch.tensor([[2, 1, 0]]),
        response_mask=torch.tensor([[True, True, False]]),
        evidence=evidence,
        top_k=top_k,
    )


def _oracle(engine, evidence):
    return VLLMTeacherOracle(
        engine,
        teacher_id="teacher",
        teacher_revision="teacher-revision",
        tokenizer=_Tokenizer({"a": 0, "b": 1, "c": 2}),
        evidence_kind=evidence,
    )


def test_tokenizer_vocabulary_fingerprint_depends_on_token_id_semantics():
    baseline = tokenizer_vocabulary_fingerprint(_Tokenizer({"a": 0, "b": 1}))

    assert tokenizer_vocabulary_fingerprint(_Tokenizer({"b": 1, "a": 0})) == baseline
    assert tokenizer_vocabulary_fingerprint(_Tokenizer({"a": 1, "b": 0})) != baseline


@pytest.mark.asyncio
async def test_vllm_teacher_oracle_extracts_exact_chosen_tokens_and_masks_padding():
    engine = _Engine([[None, {1: -0.1}, {2: -0.2, 0: -1.2}, {1: -0.3, 2: -0.7}, {0: -0.4}]])
    oracle = _oracle(engine, TeacherEvidenceKind.CHOSEN_TOKEN)

    evidence = await oracle.score(_request(TeacherEvidenceKind.CHOSEN_TOKEN))

    assert isinstance(evidence, ChosenTokenTeacherEvidence)
    torch.testing.assert_close(evidence.chosen_logprobs, torch.tensor([[-0.2, -0.3, torch.nan]]), equal_nan=True)
    assert engine.requests[0]["prompt_token_ids"] == [[0, 1, 2, 1]]
    assert engine.requests[0]["sampling_params"]["prompt_logprobs"] == 1


@pytest.mark.asyncio
async def test_vllm_teacher_oracle_returns_sorted_sparse_distribution_and_retained_mass():
    engine = _Engine(
        [
            [
                None,
                {1: math.log(0.9)},
                {2: math.log(0.6), 1: math.log(0.25), 0: math.log(0.1)},
                {1: math.log(0.5), 2: math.log(0.25), 0: math.log(0.1)},
                {0: math.log(0.9)},
            ]
        ]
    )
    oracle = _oracle(engine, TeacherEvidenceKind.TOPK_DISTRIBUTION)

    evidence = await oracle.score(_request(TeacherEvidenceKind.TOPK_DISTRIBUTION, top_k=2))

    assert isinstance(evidence, TopKTeacherEvidence)
    torch.testing.assert_close(evidence.topk_indices, torch.tensor([[[2, 1], [1, 2], [-1, -1]]]))
    torch.testing.assert_close(
        evidence.topk_logprobs,
        torch.tensor(
            [
                [
                    [math.log(0.6), math.log(0.25)],
                    [math.log(0.5), math.log(0.25)],
                    [torch.nan, torch.nan],
                ]
            ]
        ),
        equal_nan=True,
    )
    torch.testing.assert_close(evidence.retained_mass[0, :2], torch.tensor([0.85, 0.75]))
    assert torch.isnan(evidence.retained_mass[0, 2])


@pytest.mark.asyncio
async def test_vllm_teacher_oracle_rejects_missing_prompt_scores_and_owns_teardown():
    engine = _Engine([[None, {1: -0.1}, None, {1: -0.3}]])
    oracle = _oracle(engine, TeacherEvidenceKind.CHOSEN_TOKEN)

    with pytest.raises(ValueError, match="no prompt logprobs"):
        await oracle.score(_request(TeacherEvidenceKind.CHOSEN_TOKEN))

    await oracle.close()
    await oracle.close()
    assert engine.teardown_calls == 1
