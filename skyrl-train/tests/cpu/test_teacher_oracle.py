import asyncio

import pytest
import torch

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.distillation import ChosenTokenTeacherEvidence, TeacherScoreRequest
from skyrl_train.teacher_oracle import (
    TeacherCapabilities,
    TeacherOracleOwner,
    ValidatedTeacherOracle,
)


def _request(*, tokenizer_fingerprint: str = "sha256:shared-tokenizer") -> TeacherScoreRequest:
    return TeacherScoreRequest(
        trajectory_ids=("trajectory-0", "trajectory-1"),
        route_ids=("math", "code"),
        teacher_id="teacher-a",
        tokenizer_fingerprint=tokenizer_fingerprint,
        plan_version="routing-v3",
        prompt_token_ids=torch.tensor([[0, 11], [12, 13]]),
        prompt_mask=torch.tensor([[False, True], [True, True]]),
        response_token_ids=torch.tensor([[21, 22], [31, 0]]),
        response_mask=torch.tensor([[True, True], [True, False]]),
        evidence=TeacherEvidenceKind.CHOSEN_TOKEN,
    )


class DeterministicTeacherService:
    def __init__(
        self,
        *,
        tokenizer_fingerprint: str = "sha256:shared-tokenizer",
        teacher_revision: str = "teacher-revision-7",
        max_sequence_length: int = 8,
        supports_prompt_token_scoring: bool = True,
    ) -> None:
        self.capabilities = TeacherCapabilities(
            teacher_id="teacher-a",
            teacher_revision=teacher_revision,
            tokenizer_fingerprint=tokenizer_fingerprint,
            evidence_kinds=frozenset({TeacherEvidenceKind.CHOSEN_TOKEN}),
            max_sequence_length=max_sequence_length,
            supports_prompt_token_scoring=supports_prompt_token_scoring,
            max_concurrency=2,
        )
        self.requests: list[TeacherScoreRequest] = []
        self.close_count = 0

    async def score(self, request: TeacherScoreRequest) -> ChosenTokenTeacherEvidence:
        self.requests.append(request)
        chosen_logprobs = -request.response_token_ids.to(torch.float32) / 100
        chosen_logprobs = chosen_logprobs.masked_fill(~request.response_mask, torch.nan)
        return ChosenTokenTeacherEvidence(
            trajectory_ids=request.trajectory_ids,
            route_ids=request.route_ids,
            teacher_id=request.teacher_id,
            teacher_revision=self.capabilities.teacher_revision,
            plan_version=request.plan_version,
            valid_mask=request.response_mask.clone(),
            chosen_logprobs=chosen_logprobs,
        )

    async def close(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_validated_teacher_oracle_scores_exact_student_tokens():
    service = DeterministicTeacherService()
    oracle = ValidatedTeacherOracle(service)

    evidence = await oracle.score(_request())

    assert evidence.trajectory_ids == ("trajectory-0", "trajectory-1")
    assert evidence.plan_version == "routing-v3"
    torch.testing.assert_close(
        evidence.chosen_logprobs,
        torch.tensor([[-0.21, -0.22], [-0.31, torch.nan]]),
        equal_nan=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_kwargs", "request_kwargs", "message"),
    [
        ({}, {"tokenizer_fingerprint": "sha256:different-tokenizer"}, "tokenizer fingerprint"),
        ({"max_sequence_length": 2}, {}, "maximum sequence length"),
        ({"supports_prompt_token_scoring": False}, {}, "cannot score prompt tokens"),
    ],
)
async def test_validated_teacher_oracle_rejects_incompatible_request_before_scoring(
    service_kwargs, request_kwargs, message
):
    service = DeterministicTeacherService(**service_kwargs)
    oracle = ValidatedTeacherOracle(service)

    with pytest.raises(ValueError, match=message):
        await oracle.score(_request(**request_kwargs))

    assert service.requests == []


@pytest.mark.asyncio
async def test_validated_teacher_oracle_rejects_wrong_service_revision():
    class WrongRevisionService(DeterministicTeacherService):
        async def score(self, request: TeacherScoreRequest) -> ChosenTokenTeacherEvidence:
            evidence = await super().score(request)
            return ChosenTokenTeacherEvidence(
                trajectory_ids=evidence.trajectory_ids,
                route_ids=evidence.route_ids,
                teacher_id=evidence.teacher_id,
                teacher_revision="unexpected-revision",
                plan_version=evidence.plan_version,
                valid_mask=evidence.valid_mask,
                chosen_logprobs=evidence.chosen_logprobs,
            )

    oracle = ValidatedTeacherOracle(WrongRevisionService())

    with pytest.raises(ValueError, match="negotiated teacher revision"):
        await oracle.score(_request())


@pytest.mark.asyncio
async def test_teacher_oracle_owner_closes_started_services_after_partial_startup():
    first = DeterministicTeacherService()

    async def start_first():
        return first

    async def fail_second():
        raise RuntimeError("service startup failed")

    with pytest.raises(RuntimeError, match="service startup failed"):
        await TeacherOracleOwner.create({"teacher-a": start_first, "teacher-b": fail_second})

    assert first.close_count == 1


@pytest.mark.asyncio
async def test_teacher_oracle_owner_closes_services_once():
    service = DeterministicTeacherService()

    async def start_service():
        return service

    owner = await TeacherOracleOwner.create({"teacher-a": start_service})
    evidence = await owner.score("teacher-a", _request())
    await owner.close()
    await owner.close()

    assert evidence.teacher_revision == "teacher-revision-7"
    assert service.close_count == 1


@pytest.mark.asyncio
async def test_validated_teacher_oracle_drains_scoring_before_close():
    class BlockingTeacherService(DeterministicTeacherService):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = False

        async def score(self, request: TeacherScoreRequest) -> ChosenTokenTeacherEvidence:
            self.started.set()
            await self.release.wait()
            assert not self.closed
            return await super().score(request)

        async def close(self) -> None:
            self.closed = True
            await super().close()

    service = BlockingTeacherService()
    oracle = ValidatedTeacherOracle(service)
    score_task = asyncio.create_task(oracle.score(_request()))
    await service.started.wait()

    close_task = asyncio.create_task(oracle.close())
    _, pending = await asyncio.wait({close_task}, timeout=0)
    assert pending == {close_task}

    service.release.set()
    evidence, _ = await asyncio.gather(score_task, close_task)
    assert evidence.teacher_revision == "teacher-revision-7"
    assert service.closed


@pytest.mark.asyncio
async def test_validated_teacher_oracle_enforces_advertised_concurrency():
    class CapacityTeacherService(DeterministicTeacherService):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.peak_active = 0
            self.capacity_reached = asyncio.Event()
            self.release = asyncio.Event()

        async def score(self, request: TeacherScoreRequest) -> ChosenTokenTeacherEvidence:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            if self.active == self.capabilities.max_concurrency:
                self.capacity_reached.set()
            await self.release.wait()
            evidence = await super().score(request)
            self.active -= 1
            return evidence

    service = CapacityTeacherService()
    oracle = ValidatedTeacherOracle(service)
    score_tasks = [asyncio.create_task(oracle.score(_request())) for _ in range(3)]
    await service.capacity_reached.wait()

    assert service.peak_active == 2
    assert len(service.requests) == 0

    service.release.set()
    evidence = await asyncio.gather(*score_tasks)
    assert len(evidence) == 3
    assert service.peak_active == 2
