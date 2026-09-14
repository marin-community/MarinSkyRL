"""Build the first production teacher runtime from a compiled local-vLLM plan."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from marinskyrl.distillation import (
    DistillationPlan,
    LocalInferenceTeacherSpec,
    compile_distillation_plan_from_config,
    validate_distillation_runtime_support,
)
from skyrl_train.distillation_runtime import SyncDistillationRuntime
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
from skyrl_train.teacher_oracle import TeacherEndpoint, TeacherEndpointPool, TeacherOracleOwner
from skyrl_train.tokenizer import create_tokenizer


@dataclass(frozen=True)
class PreparedSyncDistillationRuntime:
    """Validated tokenizer and plan inputs prepared before Ray actor allocation."""

    plan: DistillationPlan
    teacher: LocalInferenceTeacherSpec
    teacher_tokenizer: PreTrainedTokenizerBase
    student_fingerprint: str


async def _close_engines(engines: Sequence[InferenceEngineInterface]) -> None:
    errors: list[BaseException] = []
    for engine in reversed(engines):
        try:
            await close_owned_inference_engine(engine)
        except BaseException as error:
            errors.append(error)
    if errors:
        raise BaseExceptionGroup("local teacher engine cleanup failed", errors)


def prepare_sync_distillation_runtime(
    cfg: DictConfig,
    student_tokenizer: PreTrainedTokenizerBase,
) -> PreparedSyncDistillationRuntime | None:
    """Validate the fixed local teacher before any training actors are allocated."""
    plan = compile_distillation_plan_from_config(cfg)
    validate_distillation_runtime_support(plan)
    if plan is None:
        return None

    teacher = plan.teachers[0]
    assert isinstance(teacher, LocalInferenceTeacherSpec)
    assert teacher.resources is not None

    teacher_tokenizer = create_tokenizer(
        teacher.model.path,
        revision=teacher.model.revision,
        disable_fast_tokenizer=bool(cfg.trainer.disable_fast_tokenizer),
    )
    student_fingerprint = tokenizer_vocabulary_fingerprint(student_tokenizer)
    teacher_fingerprint = tokenizer_vocabulary_fingerprint(teacher_tokenizer)
    if teacher_fingerprint != student_fingerprint:
        raise ValueError(
            f"teacher {teacher.id!r} tokenizer vocabulary does not match the policy tokenizer; "
            "vocabulary-level distillation requires identical token-ID semantics"
        )

    return PreparedSyncDistillationRuntime(plan, teacher, teacher_tokenizer, student_fingerprint)


async def start_sync_distillation_runtime(
    cfg: DictConfig,
    prepared: PreparedSyncDistillationRuntime | None,
) -> SyncDistillationRuntime | None:
    """Allocate the prepared local-vLLM teacher pool and transfer its ownership."""
    if prepared is None:
        return None
    plan = prepared.plan
    teacher = prepared.teacher
    teacher_tokenizer = prepared.teacher_tokenizer
    student_fingerprint = prepared.student_fingerprint

    resources = teacher.resources
    total_gpus = resources.num_nodes * resources.gpus_per_node
    engine_count = total_gpus // resources.tensor_parallel_size
    max_logprobs = teacher.top_k if teacher.top_k is not None else 1
    engine_init_kwargs = dict(OmegaConf.to_container(cfg.generator.engine_init_kwargs, resolve=True))
    engine_init_kwargs.pop("openai_sampling_params", None)
    engine_init_kwargs["revision"] = teacher.model.revision
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
        pool = TeacherEndpointPool(endpoints)
        owner = TeacherOracleOwner.from_oracles({teacher.id: pool})
    except BaseException as startup_error:
        try:
            await _close_engines(engines)
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "local teacher startup and cleanup failed",
                [startup_error, cleanup_error],
            )
        raise
    return SyncDistillationRuntime(plan, owner, tokenizer_fingerprints={teacher.id: student_fingerprint})
