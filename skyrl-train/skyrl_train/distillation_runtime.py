"""Trainer-facing orchestration for configured distillation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch

from marinskyrl.distillation import DistillationPlan
from skyrl_train.distillation_adapters import (
    AsyncRoutedTeacherScoreTicket,
    AsyncTeacherQueueLimits,
    FullyAsyncRayPPOTrainerDistillationAdapter,
    RayPPOTrainerDistillationAdapter,
    RoutedScoredDistillationBatch,
    RoutedTeacherScoringWork,
    TeacherEvidenceCoordinator,
    build_routed_teacher_scoring_work,
)
from skyrl_train.distillation import INVALID_TOPK_INDEX, SampledReverseKLInput, SparseForwardKLInput
from skyrl_train.teacher_oracle import TeacherOracleCollection
from skyrl_train.teacher_routing import PlanTeacherRouter, route_trajectory_batch
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import TrajectoryBatch


_ForwardResult = TypeVar("_ForwardResult", bound=TrainingInputBatch)


class _RoutedDistillationPlanner:
    """Compile trajectory batches into backend-neutral, routed teacher work."""

    def __init__(self, plan: DistillationPlan, *, tokenizer_fingerprints: dict[str, str]) -> None:
        self._router = PlanTeacherRouter(plan)
        self._tokenizer_fingerprints = dict(tokenizer_fingerprints)
        self._top_k_by_teacher = {teacher.id: teacher.top_k for teacher in plan.teachers if teacher.top_k is not None}
        self._default_route_key = plan.routing.routes[0].key if len(plan.routing.routes) == 1 else None

    def build_work(self, trajectory_batch: TrajectoryBatch) -> RoutedTeacherScoringWork:
        route_keys = trajectory_batch.get("teacher_route_keys")
        if route_keys is None:
            if self._default_route_key is None:
                raise ValueError("a multi-route distillation plan requires teacher_route_keys on the trajectory batch")
            route_keys = [self._default_route_key] * len(trajectory_batch["response_ids"])
        routed = route_trajectory_batch(
            trajectory_batch,
            route_keys=tuple(route_keys),
            router=self._router,
        )
        return build_routed_teacher_scoring_work(
            routed,
            tokenizer_fingerprints=self._tokenizer_fingerprints,
            top_k_by_teacher=self._top_k_by_teacher,
        )


class SyncDistillationRuntime:
    """Feed the synchronous trainer without coupling it to an oracle backend."""

    def __init__(
        self,
        plan: DistillationPlan,
        oracles: TeacherOracleCollection,
        *,
        tokenizer_fingerprints: dict[str, str],
    ) -> None:
        self._planner = _RoutedDistillationPlanner(plan, tokenizer_fingerprints=tokenizer_fingerprints)
        self._adapter = RayPPOTrainerDistillationAdapter.from_oracles(oracles)

    async def score_while_model_forwarding(
        self,
        trajectory_batch: TrajectoryBatch,
        model_forward: Callable[[], _ForwardResult],
    ) -> tuple[_ForwardResult, RoutedScoredDistillationBatch]:
        work = self._planner.build_work(trajectory_batch)
        return await self._adapter.score_routed_while_model_forwarding(work, model_forward)

    async def close(self) -> None:
        await self._adapter.close()


class AsyncDistillationRuntime:
    """Score admitted rollout groups ahead of fully-async learner batch assembly."""

    def __init__(
        self,
        plan: DistillationPlan,
        oracles: TeacherOracleCollection,
        *,
        tokenizer_fingerprints: dict[str, str],
        teacher_limits: dict[str, AsyncTeacherQueueLimits],
    ) -> None:
        self._planner = _RoutedDistillationPlanner(plan, tokenizer_fingerprints=tokenizer_fingerprints)
        self._adapter = FullyAsyncRayPPOTrainerDistillationAdapter(
            coordinator=TeacherEvidenceCoordinator(oracles),
            teacher_limits=teacher_limits,
        )

    async def start(self) -> None:
        await self._adapter.start()

    async def submit_before_batch_assembly(
        self,
        trajectory_batch: TrajectoryBatch,
    ) -> AsyncRoutedTeacherScoreTicket:
        """Route and enqueue one admitted, optimization-selected rollout group."""
        work = self._planner.build_work(trajectory_batch)
        return await self._adapter.submit_routed_before_batch_assembly(work)

    def attach_to_training_input(
        self,
        training_input: TrainingInputBatch,
        scored_groups: Sequence[RoutedScoredDistillationBatch],
    ) -> None:
        """Attach scored groups in admission order, including safe learner padding rows."""
        if not scored_groups:
            raise ValueError("fully-async distillation requires at least one scored group")
        response_shape = training_input["response_mask"].shape
        row_count = sum(len(scored.trajectory_ids) for scored in scored_groups)
        pad_size = int(training_input.metadata.get("pad_size", 0))
        if row_count + pad_size != response_shape[0]:
            raise ValueError(
                "scored distillation rows must match the unpadded learner batch: "
                f"scored={row_count}, padding={pad_size}, learner={response_shape[0]}"
            )
        inputs = [scored.distillation for scored in scored_groups]
        first = inputs[0]
        shape = tuple(response_shape)
        valid_mask = torch.zeros(shape, dtype=torch.bool)
        loss_weights = torch.zeros(shape, dtype=torch.float32)
        offset = 0

        if isinstance(first, SampledReverseKLInput):
            if not all(isinstance(item, SampledReverseKLInput) for item in inputs):
                raise ValueError("scored groups must use one distillation objective kind")
            teacher_logprobs = torch.full(shape, torch.nan, dtype=torch.float32)
            for item in inputs:
                rows, width = item.valid_mask.shape
                if width > shape[1]:
                    raise ValueError("teacher evidence is wider than the learner response batch")
                target = slice(offset, offset + rows)
                valid_mask[target, :width] = item.valid_mask
                loss_weights[target, :width] = item.loss_weights
                teacher_logprobs[target, :width] = item.teacher_action_log_probs
                offset += rows
            training_input.update(SampledReverseKLInput(teacher_logprobs, valid_mask, loss_weights).training_tensors())
        else:
            if not isinstance(first, SparseForwardKLInput) or not all(
                isinstance(item, SparseForwardKLInput) for item in inputs
            ):
                raise ValueError("scored groups must use one distillation objective kind")
            top_k_values = {item.teacher_topk_indices.shape[-1] for item in inputs}
            if len(top_k_values) != 1:
                raise ValueError("scored groups must use one teacher top-K width")
            top_k = top_k_values.pop()
            indices = torch.full((*shape, top_k), INVALID_TOPK_INDEX, dtype=torch.long)
            logprobs = torch.full((*shape, top_k), torch.nan, dtype=torch.float32)
            retained_mass = torch.full(shape, torch.nan, dtype=torch.float32)
            for item in inputs:
                rows, width = item.valid_mask.shape
                if width > shape[1]:
                    raise ValueError("teacher evidence is wider than the learner response batch")
                target = slice(offset, offset + rows)
                valid_mask[target, :width] = item.valid_mask
                loss_weights[target, :width] = item.loss_weights
                indices[target, :width] = item.teacher_topk_indices
                logprobs[target, :width] = item.teacher_topk_logprobs
                retained_mass[target, :width] = item.retained_mass
                offset += rows
            training_input.update(
                SparseForwardKLInput(indices, logprobs, retained_mass, valid_mask, loss_weights).training_tensors()
            )

        training_input.metadata["distillation_plan_versions"] = tuple(scored.plan_version for scored in scored_groups)
        training_input.metadata["distillation_teacher_revisions"] = tuple(
            revision for scored in scored_groups for revision in scored.teacher_revisions
        )
        training_input.metadata["distillation_route_ids"] = tuple(
            route.route_id for scored in scored_groups for route in scored.routes
        )

    async def close(self) -> None:
        await self._adapter.close()
