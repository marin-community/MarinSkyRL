"""Build one transport-neutral teacher fleet for both trainer regimes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable, Sequence
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from marinskyrl.distillation import (
    DistillationPlan,
    LocalInferenceTeacherSpec,
    OpenAICompatibleTeacherSpec,
    TeacherPlacement,
    compile_distillation_plan_from_config,
    validate_distillation_runtime_support,
)
from rigging.secrets import resolve_secret_spec
from skyrl_train.distillation_adapters import AsyncTeacherQueueLimits
from skyrl_train.distillation_runtime import AsyncDistillationRuntime, SyncDistillationRuntime
from skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl_train.inference_engines.configuration import (
    InferenceEngineRoleConfig,
    inference_engine_kwargs_from_config,
)
from skyrl_train.inference_engines.openai_teacher_oracle import OpenAICompatibleTeacherOracle
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.vllm_teacher_oracle import (
    VLLMTeacherOracle,
    close_owned_inference_engine,
    tokenizer_vocabulary_fingerprint,
)
from skyrl_train.teacher_oracle import (
    RotatingTeacherOracleOwner,
    TeacherEndpoint,
    TeacherEndpointPool,
    TeacherOracleFactory,
    TeacherOracleFleet,
    TeacherOracleOwner,
)
from skyrl_train.tokenizer import create_tokenizer


@dataclass(frozen=True)
class PreparedLocalTeacher:
    """One local teacher whose tokenizer contract was validated before allocation."""

    spec: LocalInferenceTeacherSpec
    tokenizer: PreTrainedTokenizerBase


@dataclass(frozen=True)
class PreparedDistillationRuntime:
    """Validated plan inputs prepared before policy or teacher allocation."""

    plan: DistillationPlan
    local_teachers: tuple[PreparedLocalTeacher, ...]
    external_teachers: tuple[OpenAICompatibleTeacherSpec, ...]
    student_fingerprint: str


@dataclass(frozen=True)
class PreparedAsyncDistillationRuntime(PreparedDistillationRuntime):
    """Teacher inputs plus validated fully-async queue limits."""

    teacher_limits: AsyncTeacherQueueLimits


async def _close_engines(engines: Sequence[InferenceEngineInterface]) -> None:
    await _gather_cleanup(
        (close_owned_inference_engine(engine) for engine in reversed(engines)),
        message="local teacher engine cleanup failed",
    )


async def _gather_cleanup(awaitables: Iterable[Awaitable[None]], *, message: str) -> None:
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise BaseExceptionGroup(message, errors)


def _validated_distillation_plan(cfg: DictConfig) -> DistillationPlan | None:
    plan = compile_distillation_plan_from_config(cfg)
    validate_distillation_runtime_support(plan)
    return plan


def _prepare_distillation_runtime(
    cfg: DictConfig,
    student_tokenizer: PreTrainedTokenizerBase,
    plan: DistillationPlan,
) -> PreparedDistillationRuntime:
    student_fingerprint = tokenizer_vocabulary_fingerprint(student_tokenizer)
    prepared_teachers: list[PreparedLocalTeacher] = []
    external_teachers: list[OpenAICompatibleTeacherSpec] = []
    for teacher in plan.teachers:
        if isinstance(teacher, OpenAICompatibleTeacherSpec):
            if teacher.tokenizer_fingerprint != student_fingerprint:
                raise ValueError(
                    f"teacher {teacher.id!r} tokenizer fingerprint does not match the policy tokenizer; "
                    "vocabulary-level distillation requires identical token-ID semantics"
                )
            external_teachers.append(teacher)
            continue
        assert isinstance(teacher, LocalInferenceTeacherSpec)
        teacher_tokenizer = create_tokenizer(
            teacher.model.path,
            revision=teacher.model.revision,
            disable_fast_tokenizer=bool(cfg.trainer.disable_fast_tokenizer),
        )
        teacher_fingerprint = tokenizer_vocabulary_fingerprint(teacher_tokenizer)
        if teacher_fingerprint != student_fingerprint:
            raise ValueError(
                f"teacher {teacher.id!r} tokenizer vocabulary does not match the policy tokenizer; "
                "vocabulary-level distillation requires identical token-ID semantics"
            )
        prepared_teachers.append(PreparedLocalTeacher(teacher, teacher_tokenizer))

    return PreparedDistillationRuntime(
        plan,
        tuple(prepared_teachers),
        tuple(external_teachers),
        student_fingerprint,
    )


def prepare_distillation_runtime(
    cfg: DictConfig,
    student_tokenizer: PreTrainedTokenizerBase,
) -> PreparedDistillationRuntime | None:
    """Validate all teachers for synchronous training before actor allocation."""
    plan = _validated_distillation_plan(cfg)
    if plan is None:
        return None
    return _prepare_distillation_runtime(cfg, student_tokenizer, plan)


def prepare_async_distillation_runtime(
    cfg: DictConfig,
    student_tokenizer: PreTrainedTokenizerBase,
) -> PreparedAsyncDistillationRuntime | None:
    """Validate all teachers and async queue limits before actor allocation."""
    plan = _validated_distillation_plan(cfg)
    if plan is None:
        return None
    scoring = cfg.trainer.fully_async.teacher_scoring
    teacher_limits = AsyncTeacherQueueLimits(
        max_queued=scoring.max_queued_per_teacher,
        workers=scoring.workers_per_teacher,
    )
    prepared = _prepare_distillation_runtime(cfg, student_tokenizer, plan)
    assert prepared is not None
    return PreparedAsyncDistillationRuntime(
        prepared.plan,
        prepared.local_teachers,
        prepared.external_teachers,
        prepared.student_fingerprint,
        teacher_limits,
    )


async def _start_local_teacher_pool(
    cfg: DictConfig,
    prepared: PreparedLocalTeacher,
) -> TeacherEndpointPool:
    teacher = prepared.spec
    teacher_tokenizer = prepared.tokenizer
    assert teacher.resources is not None

    resources = teacher.resources
    total_gpus = resources.num_nodes * resources.gpus_per_node
    engine_count = total_gpus // resources.tensor_parallel_size
    max_logprobs = teacher.top_k if teacher.top_k is not None else 1
    engine_init_kwargs = dict(OmegaConf.to_container(cfg.generator.engine_init_kwargs, resolve=True))
    engine_init_kwargs.pop("openai_sampling_params", None)
    engine_init_kwargs["revision"] = teacher.model.revision
    engine_init_kwargs["served_model_name"] = teacher.model.path
    role = InferenceEngineRoleConfig(
        pretrain=teacher.model.path,
        backend=teacher.backend,
        num_inference_engines=engine_count,
        tensor_parallel_size=resources.tensor_parallel_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1,
        decode_context_parallel_size=1,
        shared_pg=None,
        inference_engine_enable_sleep=False,
        max_logprobs=max_logprobs,
    )
    engine_kwargs = inference_engine_kwargs_from_config(
        cfg,
        tokenizer=teacher_tokenizer,
        role=role,
        engine_init_kwargs=engine_init_kwargs,
    )
    engines = create_ray_wrapped_inference_engines(**engine_kwargs)
    try:
        endpoints = tuple(
            TeacherEndpoint(
                endpoint_id=f"{teacher.id}-{index}",
                oracle=VLLMTeacherOracle(
                    engine,
                    teacher_id=teacher.id,
                    teacher_revision=teacher.model.revision,
                    tokenizer=teacher_tokenizer,
                    evidence_kind=teacher.evidence,
                ),
            )
            for index, engine in enumerate(engines)
        )
        return TeacherEndpointPool(endpoints)
    except BaseException as startup_error:
        try:
            await _close_engines(engines)
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "local teacher startup and cleanup failed",
                [startup_error, cleanup_error],
            )
        raise


def _teacher_factory(cfg: DictConfig, prepared: PreparedLocalTeacher) -> TeacherOracleFactory:
    async def start() -> TeacherEndpointPool:
        return await _start_local_teacher_pool(cfg, prepared)

    return start


async def start_sync_distillation_runtime(
    cfg: DictConfig,
    prepared: PreparedDistillationRuntime | None,
) -> SyncDistillationRuntime | None:
    """Allocate pinned teachers eagerly and rotating teachers on demand."""
    if prepared is None:
        return None

    fleet = await _start_teacher_fleet(cfg, prepared)
    return SyncDistillationRuntime(
        prepared.plan,
        fleet,
        tokenizer_fingerprints={teacher.id: prepared.student_fingerprint for teacher in prepared.plan.teachers},
    )


async def start_async_distillation_runtime(
    cfg: DictConfig,
    prepared: PreparedAsyncDistillationRuntime | None,
) -> AsyncDistillationRuntime | None:
    """Allocate the shared teacher fleet and bounded fully-async score queues."""
    if prepared is None:
        return None

    fleet = await _start_teacher_fleet(cfg, prepared)
    runtime = AsyncDistillationRuntime(
        prepared.plan,
        fleet,
        tokenizer_fingerprints={teacher.id: prepared.student_fingerprint for teacher in prepared.plan.teachers},
        teacher_limits={teacher.id: prepared.teacher_limits for teacher in prepared.plan.teachers},
    )
    return runtime


async def _start_teacher_fleet(
    cfg: DictConfig,
    prepared: PreparedDistillationRuntime,
) -> TeacherOracleFleet:
    """Start one mixed teacher fleet shared by sync and fully-async schedulers."""

    pinned = tuple(teacher for teacher in prepared.local_teachers if teacher.spec.placement is TeacherPlacement.PINNED)
    rotating = tuple(
        teacher for teacher in prepared.local_teachers if teacher.spec.placement is TeacherPlacement.ROTATING
    )
    fixed_factories = {teacher.spec.id: _teacher_factory(cfg, teacher) for teacher in pinned}
    fixed_factories.update({teacher.id: _external_teacher_factory(teacher) for teacher in prepared.external_teachers})
    fixed_owner: TeacherOracleOwner | None = None
    rotating_owner: RotatingTeacherOracleOwner | None = None
    try:
        if fixed_factories:
            fixed_owner = await TeacherOracleOwner.create(fixed_factories)
        if rotating:
            rotating_owner = RotatingTeacherOracleOwner(
                {teacher.spec.id: _teacher_factory(cfg, teacher) for teacher in rotating},
                max_resident=prepared.plan.residency.max_resident,
                minimum_residency_seconds=prepared.plan.residency.minimum_residency_seconds,
            )
        fleet = TeacherOracleFleet(fixed=fixed_owner, rotating=rotating_owner)
    except BaseException as startup_error:
        owners = tuple(owner for owner in (rotating_owner, fixed_owner) if owner is not None)
        try:
            await _gather_cleanup(
                (owner.close() for owner in owners),
                message="teacher fleet cleanup failed",
            )
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "teacher fleet startup and cleanup failed",
                [startup_error, cleanup_error],
            )
        raise

    return fleet


def _external_teacher_factory(teacher: OpenAICompatibleTeacherSpec) -> TeacherOracleFactory:
    async def start() -> TeacherEndpointPool:
        endpoints = []
        try:
            for index, endpoint in enumerate(teacher.endpoints):
                api_key = None
                if endpoint.auth is not None:
                    resolved_auth = await asyncio.to_thread(resolve_secret_spec, endpoint.auth)
                    api_key = resolved_auth.value
                endpoints.append(
                    TeacherEndpoint(
                        endpoint_id=f"{teacher.id}-{index}",
                        oracle=OpenAICompatibleTeacherOracle(
                            teacher=teacher,
                            endpoint=endpoint,
                            api_key=api_key,
                        ),
                    )
                )
            return TeacherEndpointPool(tuple(endpoints))
        except BaseException as startup_error:
            try:
                await _gather_cleanup(
                    (endpoint.oracle.close() for endpoint in endpoints),
                    message="external teacher cleanup failed",
                )
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "external teacher startup and cleanup failed",
                    [startup_error, cleanup_error],
                )
            raise

    return start
