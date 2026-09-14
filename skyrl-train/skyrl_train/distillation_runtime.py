"""Trainer-facing orchestration for configured distillation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from marinskyrl.distillation import DistillationPlan
from skyrl_train.distillation_adapters import (
    RayPPOTrainerDistillationAdapter,
    RoutedScoredDistillationBatch,
    build_routed_teacher_scoring_work,
)
from skyrl_train.teacher_oracle import TeacherOracleCollection
from skyrl_train.teacher_routing import PlanTeacherRouter, route_trajectory_batch
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import TrajectoryBatch


_ForwardResult = TypeVar("_ForwardResult", bound=TrainingInputBatch)


class SyncDistillationRuntime:
    """Feed the synchronous trainer without coupling it to an oracle backend."""

    def __init__(
        self,
        plan: DistillationPlan,
        oracles: TeacherOracleCollection,
        *,
        tokenizer_fingerprints: dict[str, str],
    ) -> None:
        self._router = PlanTeacherRouter(plan)
        self._adapter = RayPPOTrainerDistillationAdapter.from_oracles(oracles)
        self._tokenizer_fingerprints = dict(tokenizer_fingerprints)
        self._top_k_by_teacher = {teacher.id: teacher.top_k for teacher in plan.teachers if teacher.top_k is not None}
        self._default_route_key = plan.routing.routes[0].key if len(plan.routing.routes) == 1 else None

    async def score_while_model_forwarding(
        self,
        trajectory_batch: TrajectoryBatch,
        model_forward: Callable[[], _ForwardResult],
    ) -> tuple[_ForwardResult, RoutedScoredDistillationBatch]:
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
        work = build_routed_teacher_scoring_work(
            routed,
            tokenizer_fingerprints=self._tokenizer_fingerprints,
            top_k_by_teacher=self._top_k_by_teacher,
        )
        return await self._adapter.score_routed_while_model_forwarding(work, model_forward)

    async def close(self) -> None:
        await self._adapter.close()
