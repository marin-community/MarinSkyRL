import asyncio

import pytest
import torch

from marinskyrl.distillation import (
    DistillationObjectiveKind,
    DistillationRewardMode,
    DistillationPlan,
    OpenAICompatibleTeacherSpec,
    TeacherEndpointSpec,
    TeacherEvidenceKind,
    TeacherModelSpec,
    TeacherPlacement,
    TeacherRouteSpec,
    TeacherRoutingPlan,
    TeacherSource,
)
from skyrl_train.distillation import ChosenTokenTeacherEvidence, TeacherScoreRequest
from skyrl_train.distillation_adapters import (
    AsyncTeacherQueueLimits,
    FullyAsyncRayPPOTrainerDistillationAdapter,
    RayPPOTrainerDistillationAdapter,
    TeacherEvidenceCoordinator,
    build_routed_teacher_scoring_work,
)
from skyrl_train.distillation_runtime import SyncDistillationRuntime
from skyrl_train.teacher_oracle import (
    RotatingTeacherOracleOwner,
    TeacherCapabilities,
    TeacherEndpoint,
    TeacherEndpointPool,
    TeacherEndpointUnavailable,
    TeacherOracleFleet,
    TeacherOracleOwner,
)
from skyrl_train.teacher_routing import PlanTeacherRouter, route_trajectory_batch
from skyrl_train.trajectory_runners.types import TrajectoryID


def _teacher_spec(teacher_id: str, revision: str) -> OpenAICompatibleTeacherSpec:
    return OpenAICompatibleTeacherSpec(
        id=teacher_id,
        source=TeacherSource.OPENAI_COMPATIBLE,
        placement=TeacherPlacement.EXTERNAL,
        model=TeacherModelSpec(path=f"test/{teacher_id}", revision=revision),
        evidence=TeacherEvidenceKind.CHOSEN_TOKEN,
        endpoints=(TeacherEndpointSpec(url=f"https://{teacher_id}.example/v1", auth=None),),
    )


def _plan() -> DistillationPlan:
    return DistillationPlan(
        objective=DistillationObjectiveKind.SAMPLED_REVERSE_KL,
        coefficient=0.5,
        reward_mode=DistillationRewardMode.REPLACE,
        teachers=(_teacher_spec("math-teacher", "math-r7"), _teacher_spec("code-teacher", "code-r4")),
        routing=TeacherRoutingPlan(
            name="mopd-v1",
            revision="routes-r3",
            routes=(
                TeacherRouteSpec(key="code", teacher_id="code-teacher", weight=0.25),
                TeacherRouteSpec(key="math", teacher_id="math-teacher", weight=0.75),
            ),
        ),
    )


def _trajectory_batch():
    return {
        "trajectory_ids": [
            TrajectoryID("math-a", 0),
            TrajectoryID("code-a", 0),
            TrajectoryID("math-b", 0),
            TrajectoryID("code-b", 0),
        ],
        "prompt_token_ids": [[1], [2, 3], [4], [5, 6]],
        "response_ids": [[11, 12], [21], [31], [41, 42, 43]],
        "rewards": [1.0, 1.0, 1.0, 1.0],
        "loss_masks": [[1, 1], [1], [1], [1, 1, 1]],
        "rollout_metrics": {"batch/rows": 4.0},
    }


class RoutingTeacherService:
    def __init__(
        self,
        teacher_id: str,
        teacher_revision: str,
        *,
        release: asyncio.Event | None = None,
        completed: asyncio.Event | None = None,
        completion_order: list[str] | None = None,
    ) -> None:
        self.capabilities = TeacherCapabilities(
            teacher_id=teacher_id,
            teacher_revision=teacher_revision,
            tokenizer_fingerprint="sha256:shared-tokenizer",
            evidence_kinds=frozenset({TeacherEvidenceKind.CHOSEN_TOKEN}),
            max_sequence_length=16,
            supports_prompt_token_scoring=True,
            max_concurrency=2,
        )
        self.release = release
        self.completed = completed
        self.completion_order = completion_order
        self.requests: list[TeacherScoreRequest] = []
        self.close_count = 0

    async def score(self, request: TeacherScoreRequest) -> ChosenTokenTeacherEvidence:
        self.requests.append(request)
        if self.release is not None:
            await self.release.wait()
        if self.completion_order is not None:
            self.completion_order.append(self.capabilities.teacher_id)
        if self.completed is not None:
            self.completed.set()
        logprobs = -request.response_token_ids.to(torch.float32) / 100
        return ChosenTokenTeacherEvidence(
            trajectory_ids=request.trajectory_ids,
            route_ids=request.route_ids,
            teacher_id=request.teacher_id,
            teacher_revision=self.capabilities.teacher_revision,
            plan_version=request.plan_version,
            valid_mask=request.response_mask.clone(),
            chosen_logprobs=logprobs.masked_fill(~request.response_mask, torch.nan),
        )

    async def close(self) -> None:
        self.close_count += 1


class FailedTeacherEndpoint(RoutingTeacherService):
    async def score(self, request: TeacherScoreRequest) -> ChosenTokenTeacherEvidence:
        self.requests.append(request)
        raise TeacherEndpointUnavailable("endpoint is unavailable")


@pytest.mark.asyncio
async def test_mixed_routes_survive_out_of_order_scoring_and_endpoint_failover():
    math_release = asyncio.Event()
    code_completed = asyncio.Event()
    completion_order: list[str] = []
    failed_math = FailedTeacherEndpoint("math-teacher", "math-r7")
    healthy_math = RoutingTeacherService(
        "math-teacher",
        "math-r7",
        release=math_release,
        completion_order=completion_order,
    )
    code = RoutingTeacherService(
        "code-teacher",
        "code-r4",
        completed=code_completed,
        completion_order=completion_order,
    )
    math_pool = TeacherEndpointPool(
        (
            TeacherEndpoint("math-bad", failed_math),
            TeacherEndpoint("math-good", healthy_math),
        ),
        failure_cooldown_seconds=60,
    )

    async def start_math():
        return math_pool

    async def start_code():
        return code

    owner = await TeacherOracleOwner.create({"math-teacher": start_math, "code-teacher": start_code})
    adapter = FullyAsyncRayPPOTrainerDistillationAdapter(
        TeacherEvidenceCoordinator(owner),
        teacher_limits={
            "math-teacher": AsyncTeacherQueueLimits(max_queued=2, workers=1),
            "code-teacher": AsyncTeacherQueueLimits(max_queued=2, workers=1),
        },
    )
    await adapter.start()
    routed_batch = route_trajectory_batch(
        _trajectory_batch(),
        route_keys=("math", "code", "math", "code"),
        router=PlanTeacherRouter(_plan()),
    )
    work = build_routed_teacher_scoring_work(
        routed_batch,
        tokenizer_fingerprints={
            "math-teacher": "sha256:shared-tokenizer",
            "code-teacher": "sha256:shared-tokenizer",
        },
    )

    scoring = asyncio.create_task(adapter.score_routed_before_batch_assembly(work))
    await asyncio.wait_for(code_completed.wait(), timeout=1)
    assert completion_order == ["code-teacher"]
    math_release.set()
    scored = await scoring
    await adapter.close()

    assert [partition.teacher_id for partition in routed_batch.partitions] == ["math-teacher", "code-teacher"]
    assert routed_batch.partitions[0].original_indices == (0, 2)
    assert routed_batch.partitions[1].original_indices == (1, 3)
    assert failed_math.requests[0].trajectory_ids == ("math-a_0", "math-b_0")
    assert healthy_math.requests[0].trajectory_ids == ("math-a_0", "math-b_0")
    assert code.requests[0].trajectory_ids == ("code-a_0", "code-b_0")
    assert completion_order == ["code-teacher", "math-teacher"]
    assert scored.trajectory_ids == ("math-a_0", "code-a_0", "math-b_0", "code-b_0")
    assert tuple(route.route_id for route in scored.routes) == ("math", "code", "math", "code")
    assert tuple(route.teacher_id for route in scored.routes) == (
        "math-teacher",
        "code-teacher",
        "math-teacher",
        "code-teacher",
    )
    assert tuple(route.objective_id for route in scored.routes) == ("sampled_reverse_kl",) * 4
    assert scored.teacher_revisions == ("math-r7", "code-r4", "math-r7", "code-r4")
    assert scored.plan_version == "routes-r3"
    torch.testing.assert_close(
        scored.distillation.teacher_action_log_probs,
        torch.tensor(
            [
                [-0.11, -0.12, torch.nan],
                [-0.21, torch.nan, torch.nan],
                [-0.31, torch.nan, torch.nan],
                [-0.41, -0.42, -0.43],
            ]
        ),
        equal_nan=True,
    )
    torch.testing.assert_close(
        scored.distillation.loss_weights,
        torch.tensor(
            [
                [0.375, 0.375, 0.0],
                [0.125, 0.0, 0.0],
                [0.375, 0.0, 0.0],
                [0.125, 0.125, 0.125],
            ]
        ),
    )


@pytest.mark.asyncio
async def test_sync_adapter_fans_out_mixed_routes_during_model_forward():
    math = RoutingTeacherService("math-teacher", "math-r7")
    code = RoutingTeacherService("code-teacher", "code-r4")

    async def start_math():
        return math

    async def start_code():
        return code

    adapter = RayPPOTrainerDistillationAdapter(
        TeacherEvidenceCoordinator(
            await TeacherOracleOwner.create({"math-teacher": start_math, "code-teacher": start_code})
        )
    )
    routed_batch = route_trajectory_batch(
        _trajectory_batch(),
        route_keys=("math", "code", "math", "code"),
        router=PlanTeacherRouter(_plan()),
    )
    work = build_routed_teacher_scoring_work(
        routed_batch,
        tokenizer_fingerprints={
            "math-teacher": "sha256:shared-tokenizer",
            "code-teacher": "sha256:shared-tokenizer",
        },
    )

    forward_result, scored = await adapter.score_routed_while_model_forwarding(work, lambda: "forward-complete")
    await adapter.close()

    assert forward_result == "forward-complete"
    assert scored.trajectory_ids == routed_batch.trajectory_ids
    assert math.requests[0].trajectory_ids == ("math-a_0", "math-b_0")
    assert code.requests[0].trajectory_ids == ("code-a_0", "code-b_0")


@pytest.mark.asyncio
async def test_sync_runtime_routes_mixed_batch_from_trajectory_metadata():
    math = RoutingTeacherService("math-teacher", "math-r7")
    code = RoutingTeacherService("code-teacher", "code-r4")

    async def start_math():
        return math

    async def start_code():
        return code

    runtime = SyncDistillationRuntime(
        _plan(),
        await TeacherOracleOwner.create({"math-teacher": start_math, "code-teacher": start_code}),
        tokenizer_fingerprints={
            "math-teacher": "sha256:shared-tokenizer",
            "code-teacher": "sha256:shared-tokenizer",
        },
    )
    trajectory_batch = _trajectory_batch()
    trajectory_batch["teacher_route_keys"] = ["math", "code", "math", "code"]

    forward_result, scored = await runtime.score_while_model_forwarding(
        trajectory_batch,
        lambda: "forward-complete",
    )
    await runtime.close()

    assert forward_result == "forward-complete"
    assert tuple(route.route_id for route in scored.routes) == ("math", "code", "math", "code")
    assert math.requests[0].trajectory_ids == ("math-a_0", "math-b_0")
    assert code.requests[0].trajectory_ids == ("code-a_0", "code-b_0")


@pytest.mark.asyncio
async def test_sync_runtime_requires_explicit_routes_for_multi_route_plan():
    with pytest.raises(ValueError, match="requires teacher_route_keys"):
        runtime = SyncDistillationRuntime(
            _plan(),
            {},
            tokenizer_fingerprints={
                "math-teacher": "sha256:shared-tokenizer",
                "code-teacher": "sha256:shared-tokenizer",
            },
        )
        await runtime.score_while_model_forwarding(_trajectory_batch(), lambda: "forward-complete")


def test_router_rejects_unknown_route_before_partitioning():
    with pytest.raises(ValueError, match="unknown teacher route 'science'.*routes-r3"):
        route_trajectory_batch(
            _trajectory_batch(),
            route_keys=("math", "code", "science", "code"),
            router=PlanTeacherRouter(_plan()),
        )


@pytest.mark.asyncio
async def test_endpoint_pool_uses_pending_token_load_without_changing_logical_teacher():
    release = asyncio.Event()
    first = RoutingTeacherService("math-teacher", "math-r7", release=release)
    second = RoutingTeacherService("math-teacher", "math-r7", release=release)
    pool = TeacherEndpointPool((TeacherEndpoint("first", first), TeacherEndpoint("second", second)))
    request = (
        build_routed_teacher_scoring_work(
            route_trajectory_batch(
                _trajectory_batch(),
                route_keys=("math", "math", "math", "math"),
                router=PlanTeacherRouter(
                    DistillationPlan(
                        objective=DistillationObjectiveKind.SAMPLED_REVERSE_KL,
                        coefficient=1.0,
                        reward_mode=DistillationRewardMode.REPLACE,
                        teachers=(_teacher_spec("math-teacher", "math-r7"),),
                        routing=TeacherRoutingPlan(
                            name="math-only",
                            revision="math-routes-r1",
                            routes=(TeacherRouteSpec(key="math", teacher_id="math-teacher", weight=1.0),),
                        ),
                    )
                ),
            ),
            tokenizer_fingerprints={"math-teacher": "sha256:shared-tokenizer"},
        )
        .partitions[0]
        .work.request
    )

    first_score = asyncio.create_task(pool.score(request))
    await asyncio.sleep(0)
    second_score = asyncio.create_task(pool.score(request))
    await asyncio.sleep(0)

    assert len(first.requests) == 1
    assert len(second.requests) == 1
    release.set()
    evidence = await asyncio.gather(first_score, second_score)
    await pool.close()
    assert {result.teacher_id for result in evidence} == {"math-teacher"}


@pytest.mark.asyncio
async def test_rotating_owner_drains_before_coarse_teacher_swap():
    events: list[str] = []
    release_math = asyncio.Event()
    services: dict[str, RoutingTeacherService] = {}

    def factory(teacher_id: str, revision: str, release: asyncio.Event | None = None):
        async def start():
            events.append(f"start:{teacher_id}")
            service = RoutingTeacherService(teacher_id, revision, release=release)
            services[teacher_id] = service
            original_close = service.close

            async def close():
                events.append(f"close:{teacher_id}")
                await original_close()

            service.close = close
            return service

        return start

    owner = RotatingTeacherOracleOwner(
        {
            "math-teacher": factory("math-teacher", "math-r7", release_math),
            "code-teacher": factory("code-teacher", "code-r4"),
        },
        max_resident=1,
        minimum_residency_seconds=0,
    )
    math_request = TeacherScoreRequest(
        trajectory_ids=("math-a_0",),
        route_ids=("math",),
        teacher_id="math-teacher",
        tokenizer_fingerprint="sha256:shared-tokenizer",
        plan_version="routes-r3",
        prompt_token_ids=torch.tensor([[1]]),
        prompt_mask=torch.tensor([[True]]),
        response_token_ids=torch.tensor([[11]]),
        response_mask=torch.tensor([[True]]),
        evidence=TeacherEvidenceKind.CHOSEN_TOKEN,
    )
    code_request = TeacherScoreRequest(
        trajectory_ids=("code-a_0",),
        route_ids=("code",),
        teacher_id="code-teacher",
        tokenizer_fingerprint="sha256:shared-tokenizer",
        plan_version="routes-r3",
        prompt_token_ids=torch.tensor([[2]]),
        prompt_mask=torch.tensor([[True]]),
        response_token_ids=torch.tensor([[21]]),
        response_mask=torch.tensor([[True]]),
        evidence=TeacherEvidenceKind.CHOSEN_TOKEN,
    )

    math_score = asyncio.create_task(owner.score("math-teacher", math_request))
    await asyncio.sleep(0)
    code_score = asyncio.create_task(owner.score("code-teacher", code_request))
    await asyncio.sleep(0)
    assert events == ["start:math-teacher"]

    release_math.set()
    await math_score
    await code_score
    second_code_score = await owner.score("code-teacher", code_request)
    await owner.close()

    assert second_code_score.teacher_id == "code-teacher"
    assert events == [
        "start:math-teacher",
        "close:math-teacher",
        "start:code-teacher",
        "close:code-teacher",
    ]
    assert services["math-teacher"].close_count == 1
    assert services["code-teacher"].close_count == 1


@pytest.mark.asyncio
async def test_fixed_teacher_bypasses_rotating_residency_slots():
    fixed = RoutingTeacherService("fixed-teacher", "fixed-r1")
    rotating = RoutingTeacherService("rotating-teacher", "rotating-r1")

    async def start_fixed():
        return fixed

    async def start_rotating():
        return rotating

    fleet = TeacherOracleFleet(
        fixed=await TeacherOracleOwner.create({"fixed-teacher": start_fixed}),
        rotating=RotatingTeacherOracleOwner(
            {"rotating-teacher": start_rotating},
            max_resident=1,
            minimum_residency_seconds=60,
        ),
    )

    def request(teacher_id: str) -> TeacherScoreRequest:
        return TeacherScoreRequest(
            trajectory_ids=(f"{teacher_id}-trajectory",),
            route_ids=(teacher_id,),
            teacher_id=teacher_id,
            tokenizer_fingerprint="sha256:shared-tokenizer",
            plan_version="routes-r3",
            prompt_token_ids=torch.tensor([[1]]),
            prompt_mask=torch.tensor([[True]]),
            response_token_ids=torch.tensor([[2]]),
            response_mask=torch.tensor([[True]]),
            evidence=TeacherEvidenceKind.CHOSEN_TOKEN,
        )

    await fleet.score("rotating-teacher", request("rotating-teacher"))
    await fleet.score("fixed-teacher", request("fixed-teacher"))
    await fleet.score("rotating-teacher", request("rotating-teacher"))
    await fleet.close()

    assert len(fixed.requests) == 1
    assert len(rotating.requests) == 2
    assert fixed.close_count == 1
    assert rotating.close_count == 1
