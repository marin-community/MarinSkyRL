"""Build a production teacher runtime from a compiled local-vLLM plan."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from marinskyrl.distillation import (
    DistillationPlan,
    LocalInferenceTeacherSpec,
    TeacherPlacement,
    compile_distillation_plan_from_config,
    validate_distillation_runtime_support,
)
from skyrl_train.distillation_adapters import AsyncTeacherQueueLimits
from skyrl_train.distillation_runtime import AsyncDistillationRuntime, SyncDistillationRuntime
from skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl_train.inference_engines.configuration import (
    InferenceEngineRoleConfig,
    inference_engine_kwargs_from_config,
)
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
class PreparedLocalDistillationRuntime:
    """Validated tokenizer and plan inputs prepared before Ray actor allocation."""

    plan: DistillationPlan
    teachers: tuple[PreparedLocalTeacher, ...]
    student_fingerprint: str
    async_teacher_limits: AsyncTeacherQueueLimits | None = None


async def _close_engines(engines: Sequence[InferenceEngineInterface]) -> None:
    errors: list[BaseException] = []
    for engine in reversed(engines):
        try:
            await close_owned_inference_engine(engine)
        except BaseException as error:
            errors.append(error)
    if errors:
        raise BaseExceptionGroup("local teacher engine cleanup failed", errors)


def prepare_local_distillation_runtime(
    cfg: DictConfig,
    student_tokenizer: PreTrainedTokenizerBase,
    *,
    fully_async: bool = False,
) -> PreparedLocalDistillationRuntime | None:
    """Validate every local teacher before any training actors are allocated."""
    plan = compile_distillation_plan_from_config(cfg)
    validate_distillation_runtime_support(plan)
    if plan is None:
        return None

    async_teacher_limits = None
    if fully_async:
        scoring = cfg.trainer.fully_async.teacher_scoring
        async_teacher_limits = AsyncTeacherQueueLimits(
            max_queued=scoring.max_queued_per_teacher,
            workers=scoring.workers_per_teacher,
        )

    student_fingerprint = tokenizer_vocabulary_fingerprint(student_tokenizer)
    prepared_teachers: list[PreparedLocalTeacher] = []
    for teacher in plan.teachers:
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

    return PreparedLocalDistillationRuntime(
        plan,
        tuple(prepared_teachers),
        student_fingerprint,
        async_teacher_limits,
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
    prepared: PreparedLocalDistillationRuntime | None,
) -> SyncDistillationRuntime | None:
    """Allocate pinned teachers eagerly and rotating teachers on demand."""
    if prepared is None:
        return None

    fleet = await _start_teacher_fleet(cfg, prepared)
    return SyncDistillationRuntime(
        prepared.plan,
        fleet,
        tokenizer_fingerprints={teacher.spec.id: prepared.student_fingerprint for teacher in prepared.teachers},
    )


async def start_async_distillation_runtime(
    cfg: DictConfig,
    prepared: PreparedLocalDistillationRuntime | None,
) -> AsyncDistillationRuntime | None:
    """Allocate the shared teacher fleet and bounded fully-async score queues."""
    if prepared is None:
        return None

    fleet = await _start_teacher_fleet(cfg, prepared)
    if prepared.async_teacher_limits is None:
        raise ValueError("fully-async distillation must be prepared with fully_async=True before actor allocation")
    runtime = AsyncDistillationRuntime(
        prepared.plan,
        fleet,
        tokenizer_fingerprints={teacher.spec.id: prepared.student_fingerprint for teacher in prepared.teachers},
        teacher_limits={teacher.spec.id: prepared.async_teacher_limits for teacher in prepared.teachers},
    )
    return runtime


async def _start_teacher_fleet(
    cfg: DictConfig,
    prepared: PreparedLocalDistillationRuntime,
) -> TeacherOracleFleet:
    """Start one local teacher fleet shared by sync and fully-async schedulers."""

    pinned = tuple(teacher for teacher in prepared.teachers if teacher.spec.placement is TeacherPlacement.PINNED)
    rotating = tuple(teacher for teacher in prepared.teachers if teacher.spec.placement is TeacherPlacement.ROTATING)
    fixed_owner: TeacherOracleOwner | None = None
    rotating_owner: RotatingTeacherOracleOwner | None = None
    try:
        if pinned:
            fixed_owner = await TeacherOracleOwner.create(
                {teacher.spec.id: _teacher_factory(cfg, teacher) for teacher in pinned}
            )
        if rotating:
            rotating_owner = RotatingTeacherOracleOwner(
                {teacher.spec.id: _teacher_factory(cfg, teacher) for teacher in rotating},
                max_resident=prepared.plan.residency.max_resident,
                minimum_residency_seconds=prepared.plan.residency.minimum_residency_seconds,
            )
        fleet = TeacherOracleFleet(fixed=fixed_owner, rotating=rotating_owner)
    except BaseException as startup_error:
        owners = tuple(owner for owner in (rotating_owner, fixed_owner) if owner is not None)
        results = await asyncio.gather(*(owner.close() for owner in owners), return_exceptions=True)
        cleanup_errors = [result for result in results if isinstance(result, BaseException)]
        if cleanup_errors:
            raise BaseExceptionGroup(
                "local teacher fleet startup and cleanup failed",
                [startup_error, *cleanup_errors],
            )
        raise

    return fleet
