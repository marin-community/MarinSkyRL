import asyncio
import threading
from dataclasses import replace

import pytest
import torch

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.distillation import (
    ChosenTokenTeacherEvidence,
    SparseForwardKLInput,
    TeacherScoreRequest,
    TopKTeacherEvidence,
    prepare_sparse_forward_kl,
)
from skyrl_train.distillation_adapters import (
    AsyncTeacherQueueLimits,
    FullyAsyncRayPPOTrainerDistillationAdapter,
    RayPPOTrainerDistillationAdapter,
    RoutedTeacherScoringPartition,
    RoutedTeacherScoringWork,
    ScoredDistillationBatch,
    TeacherEvidenceCoordinator,
    TeacherScoringWork,
    build_teacher_scoring_work,
)
from skyrl_train.teacher_oracle import TeacherCapabilities, TeacherOracleOwner
from skyrl_train.teacher_routing import TeacherRoute
from skyrl_train.trajectory_runners.types import TrajectoryID


def _request(trajectory_id: str = "trajectory-0", teacher_id: str = "teacher-a") -> TeacherScoreRequest:
    return TeacherScoreRequest(
        trajectory_ids=(trajectory_id,),
        route_ids=("math",),
        teacher_id=teacher_id,
        tokenizer_fingerprint="sha256:shared-tokenizer",
        plan_version="routing-v3",
        prompt_token_ids=torch.tensor([[11, 12]]),
        prompt_mask=torch.tensor([[True, True]]),
        response_token_ids=torch.tensor([[21, 22]]),
        response_mask=torch.tensor([[True, True]]),
        evidence=TeacherEvidenceKind.CHOSEN_TOKEN,
    )


def _work(trajectory_id: str = "trajectory-0", teacher_id: str = "teacher-a") -> TeacherScoringWork:
    return TeacherScoringWork(
        request=_request(trajectory_id, teacher_id),
        coefficient=0.5,
        route_weights=torch.tensor([[1.0, 0.25]]),
    )


class ControlledTeacherService:
    def __init__(self, teacher_id: str = "teacher-a") -> None:
        self.capabilities = TeacherCapabilities(
            teacher_id=teacher_id,
            teacher_revision="teacher-revision-7",
            tokenizer_fingerprint="sha256:shared-tokenizer",
            evidence_kinds=frozenset({TeacherEvidenceKind.CHOSEN_TOKEN}),
            max_sequence_length=8,
            supports_prompt_token_scoring=True,
            max_concurrency=4,
        )
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.release = asyncio.Event()

    async def score(self, request: TeacherScoreRequest) -> ChosenTokenTeacherEvidence:
        await self.started.put(request.trajectory_ids[0])
        await self.release.wait()
        chosen_logprobs = torch.tensor([[-0.21, -0.22]])
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
        return None


async def _coordinator(*services: ControlledTeacherService) -> TeacherEvidenceCoordinator:
    factories = {}
    for service in services:

        async def start_service(service=service):
            return service

        factories[service.capabilities.teacher_id] = start_service
    owner = await TeacherOracleOwner.create(factories)
    return TeacherEvidenceCoordinator(owner)


@pytest.mark.asyncio
async def test_coordinator_prepares_sparse_forward_kl_from_the_same_scoring_contract():
    class TopKTeacherService:
        capabilities = TeacherCapabilities(
            teacher_id="teacher-a",
            teacher_revision="teacher-revision-8",
            tokenizer_fingerprint="sha256:shared-tokenizer",
            evidence_kinds=frozenset({TeacherEvidenceKind.TOPK_DISTRIBUTION}),
            max_sequence_length=8,
            supports_prompt_token_scoring=True,
            max_concurrency=4,
        )

        async def score(self, request: TeacherScoreRequest) -> TopKTeacherEvidence:
            return TopKTeacherEvidence(
                trajectory_ids=request.trajectory_ids,
                route_ids=request.route_ids,
                teacher_id=request.teacher_id,
                teacher_revision=self.capabilities.teacher_revision,
                plan_version=request.plan_version,
                valid_mask=request.response_mask,
                topk_indices=torch.tensor([[[1, 2], [3, 4]]]),
                topk_logprobs=torch.log(torch.tensor([[[0.7, 0.2], [0.6, 0.2]]])),
                retained_mass=torch.tensor([[0.9, 0.8]]),
            )

        async def close(self) -> None:
            return None

    service = TopKTeacherService()

    async def start_service():
        return service

    owner = await TeacherOracleOwner.create({"teacher-a": start_service})
    coordinator = TeacherEvidenceCoordinator(owner)
    request = replace(_request(), evidence=TeacherEvidenceKind.TOPK_DISTRIBUTION, top_k=2)
    scored = await coordinator.score(TeacherScoringWork(request, 0.5, torch.tensor([[1.0, 0.25]])))
    await coordinator.close()

    assert isinstance(scored.distillation, SparseForwardKLInput)
    torch.testing.assert_close(scored.distillation.retained_mass, torch.tensor([[0.9, 0.8]]))
    torch.testing.assert_close(scored.distillation.loss_weights, torch.tensor([[0.5, 0.125]]))


def test_routed_sparse_evidence_reassembles_original_row_order():
    requests = (
        replace(_request("trajectory-b", "teacher-b"), evidence=TeacherEvidenceKind.TOPK_DISTRIBUTION, top_k=2),
        replace(_request("trajectory-a", "teacher-a"), evidence=TeacherEvidenceKind.TOPK_DISTRIBUTION, top_k=2),
    )
    topk_indices = (torch.tensor([[[3, 4], [5, 6]]]), torch.tensor([[[1, 2], [7, 8]]]))
    topk_probs = (torch.tensor([[[0.6, 0.2], [0.5, 0.2]]]), torch.tensor([[[0.7, 0.2], [0.8, 0.1]]]))
    scored = []
    partitions = []
    for original_index, request, indices, probs in zip((1, 0), requests, topk_indices, topk_probs, strict=True):
        evidence = TopKTeacherEvidence(
            trajectory_ids=request.trajectory_ids,
            route_ids=request.route_ids,
            teacher_id=request.teacher_id,
            teacher_revision=f"{request.teacher_id}-revision",
            plan_version=request.plan_version,
            valid_mask=request.response_mask,
            topk_indices=indices,
            topk_logprobs=probs.log(),
            retained_mass=probs.sum(dim=-1),
        )
        work = TeacherScoringWork(request, 0.5, torch.ones(1, 2))
        partitions.append(RoutedTeacherScoringPartition((original_index,), work))
        scored.append(
            ScoredDistillationBatch(
                evidence,
                prepare_sparse_forward_kl(request, evidence, coefficient=0.5, route_weights=torch.ones(1, 2)),
            )
        )
    routed_work = RoutedTeacherScoringWork(
        trajectory_ids=("trajectory-a", "trajectory-b"),
        routes=(
            TeacherRoute("math", "teacher-a", "sparse_forward_kl", 1.0, "routing-v3"),
            TeacherRoute("math", "teacher-b", "sparse_forward_kl", 1.0, "routing-v3"),
        ),
        response_lengths=(2, 2),
        plan_version="routing-v3",
        partitions=tuple(partitions),
    )

    assembled = TeacherEvidenceCoordinator.assemble_routed(routed_work, tuple(scored))

    assert isinstance(assembled.distillation, SparseForwardKLInput)
    torch.testing.assert_close(assembled.distillation.teacher_topk_indices[0], topk_indices[1][0])
    torch.testing.assert_close(assembled.distillation.teacher_topk_indices[1], topk_indices[0][0])
    assert assembled.teacher_revisions == ("teacher-a-revision", "teacher-b-revision")


def test_teacher_scoring_work_collates_exact_rollout_coordinates_and_route_weights():
    work = build_teacher_scoring_work(
        {
            "trajectory_ids": [TrajectoryID("instance-a", 0), TrajectoryID("instance-b", 2)],
            "prompt_token_ids": [[11], [12, 13]],
            "response_ids": [[21, 22], [31]],
        },
        route_ids=("math", "code"),
        teacher_id="teacher-a",
        tokenizer_fingerprint="sha256:shared-tokenizer",
        plan_version="routing-v3",
        coefficient=0.5,
        route_weights=(1.0, 0.25),
    )

    assert work.request.trajectory_ids == ("instance-a_0", "instance-b_2")
    torch.testing.assert_close(work.request.prompt_token_ids, torch.tensor([[11, 0], [12, 13]]))
    torch.testing.assert_close(work.request.prompt_mask, torch.tensor([[True, False], [True, True]]))
    torch.testing.assert_close(work.request.response_token_ids, torch.tensor([[21, 22], [31, 0]]))
    torch.testing.assert_close(work.request.response_mask, torch.tensor([[True, True], [True, False]]))
    torch.testing.assert_close(work.route_weights, torch.tensor([[1.0, 1.0], [0.25, 0.0]]))


@pytest.mark.asyncio
async def test_sync_adapter_overlaps_teacher_score_with_model_forward():
    teacher_started = threading.Event()
    forward_started = threading.Event()

    class InterlockedTeacherService(ControlledTeacherService):
        async def score(self, request: TeacherScoreRequest) -> ChosenTokenTeacherEvidence:
            teacher_started.set()
            assert await asyncio.to_thread(forward_started.wait, 1)
            self.release.set()
            return await super().score(request)

    service = InterlockedTeacherService()
    adapter = RayPPOTrainerDistillationAdapter(await _coordinator(service))

    def model_forward() -> str:
        forward_started.set()
        assert teacher_started.wait(1)
        return "forward-complete"

    forward_result, scored = await asyncio.wait_for(
        adapter.score_while_model_forwarding(_work(), model_forward),
        timeout=2,
    )

    assert forward_result == "forward-complete"
    torch.testing.assert_close(scored.distillation.loss_weights, torch.tensor([[0.5, 0.125]]))
    await adapter.close()


@pytest.mark.asyncio
async def test_fully_async_adapter_scores_generated_group_before_batch_assembly():
    service = ControlledTeacherService()
    adapter = FullyAsyncRayPPOTrainerDistillationAdapter(
        await _coordinator(service),
        teacher_limits={"teacher-a": AsyncTeacherQueueLimits(max_queued=1, workers=1)},
    )
    await adapter.start()

    scoring = asyncio.create_task(adapter.score_before_batch_assembly(_work()))
    assert await service.started.get() == "trajectory-0"
    assert not scoring.done()

    service.release.set()
    scored = await scoring
    await adapter.close()

    assert scored.evidence.trajectory_ids == ("trajectory-0",)
    torch.testing.assert_close(scored.distillation.teacher_action_log_probs, torch.tensor([[-0.21, -0.22]]))


@pytest.mark.asyncio
async def test_fully_async_adapter_bounds_pending_teacher_work_per_teacher():
    service = ControlledTeacherService()
    adapter = FullyAsyncRayPPOTrainerDistillationAdapter(
        await _coordinator(service),
        teacher_limits={"teacher-a": AsyncTeacherQueueLimits(max_queued=1, workers=1)},
    )
    await adapter.start()

    first = await adapter.submit_before_batch_assembly(_work("trajectory-0"))
    assert await service.started.get() == "trajectory-0"
    second = await adapter.submit_before_batch_assembly(_work("trajectory-1"))
    third_submission = asyncio.create_task(adapter.submit_before_batch_assembly(_work("trajectory-2")))
    _, pending = await asyncio.wait({third_submission}, timeout=0)
    assert pending == {third_submission}

    service.release.set()
    third = await third_submission
    scored = await asyncio.gather(first.result(), second.result(), third.result())
    await adapter.close()

    assert [batch.evidence.trajectory_ids for batch in scored] == [
        ("trajectory-0",),
        ("trajectory-1",),
        ("trajectory-2",),
    ]


@pytest.mark.asyncio
async def test_fully_async_adapter_does_not_block_one_teacher_behind_another_full_queue():
    service_a = ControlledTeacherService("teacher-a")
    service_b = ControlledTeacherService("teacher-b")
    limits = AsyncTeacherQueueLimits(max_queued=1, workers=1)
    adapter = FullyAsyncRayPPOTrainerDistillationAdapter(
        await _coordinator(service_a, service_b),
        teacher_limits={"teacher-a": limits, "teacher-b": limits},
    )
    await adapter.start()

    first_a = await adapter.submit_before_batch_assembly(_work("a-0"))
    assert await service_a.started.get() == "a-0"
    second_a = await adapter.submit_before_batch_assembly(_work("a-1"))
    blocked_a = asyncio.create_task(adapter.submit_before_batch_assembly(_work("a-2")))
    _, pending = await asyncio.wait({blocked_a}, timeout=0)
    assert pending == {blocked_a}

    ticket_b = await adapter.submit_before_batch_assembly(_work("b-0", "teacher-b"))
    assert await service_b.started.get() == "b-0"

    service_a.release.set()
    service_b.release.set()
    third_a = await blocked_a
    scored = await asyncio.gather(first_a.result(), second_a.result(), third_a.result(), ticket_b.result())
    await adapter.close()

    assert [batch.evidence.teacher_id for batch in scored] == ["teacher-a", "teacher-a", "teacher-a", "teacher-b"]
