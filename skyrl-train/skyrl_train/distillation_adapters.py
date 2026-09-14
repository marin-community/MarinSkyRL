"""Trainer-regime adapters for scheduling teacher scoring."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

import torch
from torch.nn.utils.rnn import pad_sequence

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.distillation import (
    ChosenTokenTeacherEvidence,
    SampledReverseKLInput,
    TeacherScoreRequest,
    prepare_sampled_reverse_kl,
)
from skyrl_train.teacher_oracle import TeacherOracleOwner
from skyrl_train.trajectory_runners.types import TrajectoryBatch


_ForwardResult = TypeVar("_ForwardResult")


@dataclass(frozen=True)
class TeacherScoringWork:
    """One transport-neutral request plus its learner-side objective weights."""

    request: TeacherScoreRequest
    coefficient: float
    route_weights: torch.Tensor


@dataclass(frozen=True)
class ScoredDistillationBatch:
    """Validated evidence and the minimal payload consumed by the learner."""

    evidence: ChosenTokenTeacherEvidence
    distillation: SampledReverseKLInput


def _pad_token_rows(token_rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [torch.tensor(tokens, dtype=torch.long) for tokens in token_rows]
    padded = pad_sequence(rows, batch_first=True, padding_value=0)
    mask = torch.arange(padded.shape[1]).unsqueeze(0) < torch.tensor([len(tokens) for tokens in token_rows]).unsqueeze(
        1
    )
    return padded, mask


def build_teacher_scoring_work(
    trajectory_batch: TrajectoryBatch,
    *,
    route_ids: tuple[str, ...],
    teacher_id: str,
    tokenizer_fingerprint: str,
    plan_version: str,
    coefficient: float,
    route_weights: tuple[float, ...],
    evidence: TeacherEvidenceKind = TeacherEvidenceKind.CHOSEN_TOKEN,
) -> TeacherScoringWork:
    """Collate exact admitted rollout tokens into the shared oracle contract."""
    trajectory_ids = trajectory_batch.get("trajectory_ids")
    if trajectory_ids is None:
        raise ValueError("teacher scoring requires stable trajectory_ids")
    prompt_token_ids = trajectory_batch["prompt_token_ids"]
    response_token_ids = trajectory_batch["response_ids"]
    batch_size = len(trajectory_ids)
    if batch_size == 0:
        raise ValueError("teacher scoring requires at least one trajectory")
    aligned_fields = {
        "prompt_token_ids": len(prompt_token_ids),
        "response_ids": len(response_token_ids),
        "route_ids": len(route_ids),
        "route_weights": len(route_weights),
    }
    if any(length != batch_size for length in aligned_fields.values()):
        raise ValueError(f"teacher scoring fields must align with {batch_size} trajectories: {aligned_fields}")

    padded_prompts, prompt_mask = _pad_token_rows(prompt_token_ids)
    padded_responses, response_mask = _pad_token_rows(response_token_ids)
    per_trajectory_weights = torch.tensor(route_weights, dtype=torch.float32).unsqueeze(1)
    loss_weights = per_trajectory_weights.expand_as(padded_responses).masked_fill(~response_mask, 0)

    request = TeacherScoreRequest(
        trajectory_ids=tuple(trajectory_id.to_string() for trajectory_id in trajectory_ids),
        route_ids=route_ids,
        teacher_id=teacher_id,
        tokenizer_fingerprint=tokenizer_fingerprint,
        plan_version=plan_version,
        prompt_token_ids=padded_prompts,
        prompt_mask=prompt_mask,
        response_token_ids=padded_responses,
        response_mask=response_mask,
        evidence=evidence,
    )
    return TeacherScoringWork(request=request, coefficient=coefficient, route_weights=loss_weights)


class TeacherEvidenceCoordinator:
    """Share scoring and objective preparation across trainer regimes."""

    def __init__(self, oracle_owner: TeacherOracleOwner) -> None:
        self._oracle_owner = oracle_owner

    async def score(self, work: TeacherScoringWork) -> ScoredDistillationBatch:
        evidence = await self._oracle_owner.score(work.request.teacher_id, work.request)
        if not isinstance(evidence, ChosenTokenTeacherEvidence):
            raise ValueError("sampled reverse KL requires chosen-token teacher evidence")
        distillation = prepare_sampled_reverse_kl(
            work.request,
            evidence,
            coefficient=work.coefficient,
            route_weights=work.route_weights,
        )
        return ScoredDistillationBatch(evidence=evidence, distillation=distillation)

    async def close(self) -> None:
        await self._oracle_owner.close()


class RayPPOTrainerDistillationAdapter:
    """Overlap synchronous trainer model forwards with teacher scoring."""

    def __init__(self, coordinator: TeacherEvidenceCoordinator) -> None:
        self._coordinator = coordinator

    async def score_while_model_forwarding(
        self,
        work: TeacherScoringWork,
        model_forward: Callable[[], _ForwardResult],
    ) -> tuple[_ForwardResult, ScoredDistillationBatch]:
        """Run the blocking trainer forward and remote teacher score concurrently."""
        forward_task = asyncio.create_task(asyncio.to_thread(model_forward))
        score_task = asyncio.create_task(self._coordinator.score(work))
        try:
            forward_result, scored = await asyncio.gather(forward_task, score_task)
        except BaseException:
            forward_task.cancel()
            score_task.cancel()
            await asyncio.gather(forward_task, score_task, return_exceptions=True)
            raise
        return forward_result, scored

    async def close(self) -> None:
        await self._coordinator.close()


@dataclass(frozen=True)
class AsyncTeacherScoreTicket:
    """A submitted score whose result gates fully-async batch assembly."""

    _result: asyncio.Future[ScoredDistillationBatch]

    async def result(self) -> ScoredDistillationBatch:
        return await self._result


@dataclass(frozen=True)
class _QueuedTeacherScore:
    work: TeacherScoringWork
    result: asyncio.Future[ScoredDistillationBatch]


@dataclass(frozen=True)
class AsyncTeacherQueueLimits:
    """Bound queued and active score requests for one logical teacher."""

    max_queued: int
    workers: int

    def __post_init__(self) -> None:
        if self.max_queued <= 0 or self.workers <= 0:
            raise ValueError("teacher queue and worker limits must be positive")


class FullyAsyncRayPPOTrainerDistillationAdapter:
    """Bound teacher work per logical teacher before rollout batch assembly."""

    def __init__(
        self,
        coordinator: TeacherEvidenceCoordinator,
        *,
        teacher_limits: Mapping[str, AsyncTeacherQueueLimits],
    ) -> None:
        if not teacher_limits:
            raise ValueError("fully-async distillation requires at least one teacher queue")
        self._coordinator = coordinator
        self._queues = {
            teacher_id: asyncio.Queue[_QueuedTeacherScore](maxsize=limits.max_queued)
            for teacher_id, limits in teacher_limits.items()
        }
        self._worker_counts = {teacher_id: limits.workers for teacher_id, limits in teacher_limits.items()}
        self._workers: list[asyncio.Task[None]] = []
        self._accepting = False
        self._closed = False
        self._submission_locks = {teacher_id: asyncio.Lock() for teacher_id in teacher_limits}

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("fully-async distillation adapter is closed")
        if self._workers:
            return
        self._accepting = True
        self._workers = [
            asyncio.create_task(self._score_worker(queue))
            for teacher_id, queue in self._queues.items()
            for _ in range(self._worker_counts[teacher_id])
        ]

    async def submit_before_batch_assembly(self, work: TeacherScoringWork) -> AsyncTeacherScoreTicket:
        """Apply queue backpressure while admitting a generated group for scoring."""
        try:
            queue = self._queues[work.request.teacher_id]
            submission_lock = self._submission_locks[work.request.teacher_id]
        except KeyError as error:
            raise ValueError(f"no scoring queue configured for teacher {work.request.teacher_id!r}") from error
        async with submission_lock:
            if not self._accepting:
                raise RuntimeError("fully-async distillation adapter is not accepting work")
            result = asyncio.get_running_loop().create_future()
            await queue.put(_QueuedTeacherScore(work=work, result=result))
        return AsyncTeacherScoreTicket(result)

    async def score_before_batch_assembly(self, work: TeacherScoringWork) -> ScoredDistillationBatch:
        """Return only once evidence is ready to travel with a generated group."""
        ticket = await self.submit_before_batch_assembly(work)
        return await ticket.result()

    async def close(self) -> None:
        if self._closed:
            return
        self._accepting = False
        self._closed = True
        for submission_lock in self._submission_locks.values():
            async with submission_lock:
                pass
        if not self._workers:
            await self._coordinator.close()
            return
        await asyncio.gather(*(queue.join() for queue in self._queues.values()))
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        await self._coordinator.close()

    async def __aenter__(self) -> FullyAsyncRayPPOTrainerDistillationAdapter:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    async def _score_worker(
        self,
        queue: asyncio.Queue[_QueuedTeacherScore],
    ) -> None:
        while True:
            queued = await queue.get()
            try:
                scored = await self._coordinator.score(queued.work)
            except Exception as error:
                if not queued.result.done():
                    queued.result.set_exception(error)
            else:
                if not queued.result.done():
                    queued.result.set_result(scored)
            finally:
                queue.task_done()
