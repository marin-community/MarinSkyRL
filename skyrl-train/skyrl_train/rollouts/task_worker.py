"""TaskCompendium execution and projection into SkyRL training batches."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.models import TaskSpec
from rolloutengine.contracts import (
    GenerationLimitReached,
    ModelRequest,
    ModelTurn,
    RolloutContractError,
    RolloutData,
    RolloutEngine,
    RolloutFailure,
    RolloutInterrupted,
    RolloutOperation,
)
from rolloutengine.engine import ShellboxRolloutEngine
from taskcompendium.submission import AnswerFormat, SubmissionConvention
from skyrl_gym.envs.registration import EnvSpec, registry

from skyrl_train.inference_engines.base import ChatContinuation, InferenceEngineInterface, InferenceEngineInput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.rollouts.buffer import RolloutGroup, RolloutTask, RolloutWriter
from taskcompendium.importers.skyrl import GYM_INTERACTION
from skyrl_train.rollouts.group_grading import GROUP_GRADERS, GroupGrader, grade_groups
from skyrl_train.rollouts.gym_tasks import GymTaskSession, grade_result
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings, harbor_grading_failure, shape_harbor_rollouts
from skyrl_train.rollouts.workers import WorkerShard, detached_config
from skyrl_train.rollouts.finalization import finalize_trajectory_batch, propagate_data_sources
from skyrl_train.rollout_observability import rollout_phase, rollout_wait
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.trajectory_runners.model_clients import (
    DirectModelClient,
    GenerationBudgetExceededError,
    ModelClient,
    ModelServerError,
)
from skyrl_train.trajectory_runners.projections import (
    StepWiseTrajectoryProjection,
    WholeTrajectoryProjection,
)
from skyrl_train.rollouts.task_projections import (
    StepTaskProjection,
    WholeTaskProjection,
    verification_result,
)
from skyrl_train.trajectory_runners.skyrl_gym_contracts import fold_verification_results
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig
from skyrl_train.trajectory_runners.types import (
    TokenProvenance,
    TrajectoryBatch,
    TrajectoryRequestBatch,
)


async def _model_turn(
    client: ModelClient,
    request: ModelRequest,
    *,
    sampling_params: dict[str, Any],
    max_context_length: int,
    max_prompt_length: int | None,
    chat_template_kwargs: dict[str, Any],
    session_id: str,
) -> ModelTurn:
    """Convert an inference response to the common exact-token record."""
    options = dict(request.options)
    template_kwargs = {**chat_template_kwargs, **options.get("chat_template_kwargs", {})}
    if template_kwargs:
        options["chat_template_kwargs"] = template_kwargs
    if "tools" in options:
        options["tools"] = [{"type": "function", **tool["function"]} for tool in options["tools"]]
    continuation = None
    if request.assistant_message_index is not None:
        continuation = ChatContinuation(
            served_prefix_token_ids=list(request.prefix_token_ids),
            assistant_message_index=request.assistant_message_index,
        )
    try:
        with rollout_wait("model_client_await"):
            output = await client.generate(
                InferenceEngineInput(
                    prompts=[list(request.messages)],
                    chat_completion_params=[options],
                    chat_continuations=[continuation],
                    session_ids=[session_id],
                    sampling_params=sampling_params,
                    max_context_length=max_context_length,
                    max_prompt_length=max_prompt_length,
                )
            )
    except GenerationBudgetExceededError as limit:
        raise GenerationLimitReached(limit.prompt_token_ids) from limit
    if output["token_provenance"] != TokenProvenance.ENGINE:
        raise RolloutContractError("The canonical rollout engine requires exact model tokens")
    logprobs = output.get("response_logprobs")
    return ModelTurn(
        message=output["assistant_messages"][0],
        prompt_token_ids=tuple(output["prompt_ids"][0]),
        response_token_ids=tuple(output["response_ids"][0]),
        logprobs=None if logprobs is None else tuple(logprobs[0]),
        stop_reason=output["stop_reasons"][0],
        text=output.get("responses", [output["assistant_messages"][0].get("content") or ""])[0],
        metadata={
            **{
                name: output[name][0]
                for name in ("student_topk_indices", "behavior_topk_logprobs", "routed_experts")
                if output.get(name) is not None
            },
            "generation_token_budget": (output.get("generation_token_budgets") or [None])[0],
        },
    )


def _failed_rollout(
    interruption: RolloutInterrupted,
    task: TaskSpec,
    config: ErrorHandlingConfig,
    *,
    logprobs_required: bool,
) -> RolloutData:
    """Retain verified turns and safe diagnostics for the training policy."""
    error = interruption.__cause__
    assert isinstance(error, Exception)
    if interruption.operation == RolloutOperation.MODEL and not isinstance(error, (ModelServerError, TimeoutError)):
        raise error
    exception_type = type(error).__name__
    diagnostics = {}
    if isinstance(error, ModelServerError):
        if error.category == "context_overflow":
            exception_type = "ContextLengthExceededError"
        diagnostics = {
            "error_category": error.category,
            "request_id": error.request_id,
            "status_code": error.status_code,
        }
    elif isinstance(error, TimeoutError):
        exception_type = {
            RolloutOperation.ATTEMPT: "TrialTimeoutError",
            RolloutOperation.START: "EnvironmentStartTimeoutError",
            RolloutOperation.GRADE: "VerifierTimeoutError",
        }.get(interruption.operation, "AgentTimeoutError")
    rollout = interruption.rollout
    recover = (
        bool(rollout.steps)
        and (
            interruption.operation != RolloutOperation.GRADE
            or (isinstance(error, TimeoutError) and config.preserve_logprobs_on_timeout)
        )
        and (not logprobs_required or rollout.logprobs is not None)
        and (
            not isinstance(error, TimeoutError)
            or config.preserve_logprobs_on_timeout
            or (task.environment.interaction != GYM_INTERACTION and rollout.grade.status == Outcome.GRADED)
        )
        and (not isinstance(error, ModelServerError) or error.category == "context_overflow")
    )
    grade = GradeResult(Outcome.INFRA_ERROR, None, "Rollout execution failed")
    if recover and interruption.operation != RolloutOperation.GRADE and task.environment.interaction == GYM_INTERACTION:
        verifications = [
            verification_result(step.transition.grade) for step in rollout.steps if step.transition.grade is not None
        ]
        verification, _ = fold_verification_results(verifications)
        grade = grade_result(verification)
    elif recover and interruption.operation != RolloutOperation.GRADE:
        grade = rollout.grade
    if not recover:
        rollout = replace(rollout, response_token_ids=(), loss_mask=(), logprobs=(), steps=(), metrics={})
    logger.warning("Task {} interrupted during {}: {}", task.id, interruption.operation, exception_type)
    return replace(
        rollout,
        grade=grade,
        stop_reason="error",
        failure=RolloutFailure(exception_type, diagnostics),
    )


async def _run_engine_task(engine: RolloutEngine, task: TaskSpec, executor: Executor) -> RolloutData:
    """Run the blocking iterator in a worker thread and await cancellation cleanup."""
    pending = asyncio.get_running_loop().run_in_executor(
        executor, copy_context().run, list, engine.generate(iter([task]))
    )
    try:
        results = await asyncio.shield(pending)
    except asyncio.CancelledError:
        engine.cancel()
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                continue
        try:
            pending.result()
        except asyncio.CancelledError:
            pass
        raise
    return results[0]


class TaskRolloutWorker:
    """Run canonical task rollouts and submit complete groups to the buffer."""

    def __init__(
        self,
        trajectory_runner_cfg: DictConfig,
        projection: WholeTaskProjection | StepTaskProjection,
        model_client: ModelClient,
        factories: Mapping[EnvironmentKind, MachineFactory],
        command_timeout: float,
        max_env_workers: int = 0,
        harbor: HarborTaskSettings | None = None,
        concurrent_tasks: int | None = None,
        concurrent_harbor_tasks: int | None = None,
        retry_wait: Callable[[float], Awaitable[None]] = asyncio.sleep,
        group_graders: Mapping[str, GroupGrader] = GROUP_GRADERS,
    ):
        self.trajectory_runner_cfg = trajectory_runner_cfg
        self.model_client = model_client
        self.factories = factories
        self.command_timeout = command_timeout
        self.harbor = harbor
        self.retry_wait = retry_wait
        self.group_graders = group_graders
        self.task_slots = asyncio.Semaphore(concurrent_tasks) if concurrent_tasks is not None else nullcontext()
        self.harbor_task_slots = (
            nullcontext()
            if harbor is None
            else asyncio.Semaphore(
                harbor.concurrent_trials if concurrent_harbor_tasks is None else concurrent_harbor_tasks
            )
        )
        self.harbor_eval_slots = nullcontext() if harbor is None else asyncio.Semaphore(harbor.concurrent_trials)
        self.error_handling = ErrorHandlingConfig.from_mapping(trajectory_runner_cfg.get("error_handling", {}))
        self.projection = replace(
            projection,
            error_handling=self.error_handling,
            harbor_error_handling=None if harbor is None else harbor.error_handling,
        )
        self.engine_executor = ThreadPoolExecutor(max_workers=concurrent_tasks, thread_name_prefix="rollout-engine")
        self.environment_executor = (
            ThreadPoolExecutor(max_workers=max_env_workers, thread_name_prefix="task-environment")
            if max_env_workers > 0
            else None
        )
        self.trajectory_sink: RetentionSink | None = None

    async def generate(self, request: TrajectoryRequestBatch) -> list[RolloutData]:
        extras = request.get("env_extras")
        if extras is None or len(extras) != len(request["prompts"]):
            raise ValueError("Each rollout request must contain one task_spec per prompt")
        tasks = [TaskSpec.model_validate_json(row["task_spec"]) for row in extras]
        metadata = request.get("batch_metadata")
        phase = "train" if metadata is None else metadata.training_phase
        if self.harbor is not None:
            tasks = [self.harbor.task(task, phase=phase) if "harbor" in task.metadata else task for task in tasks]
        sampling = get_sampling_params_for_backend(
            self.trajectory_runner_cfg.backend, self.trajectory_runner_cfg.sampling_params
        )
        sampling.update(request.get("sampling_params") or {})
        context_limit = OmegaConf.select(self.trajectory_runner_cfg, "engine_init_kwargs.max_model_len")
        max_input_length = int(self.trajectory_runner_cfg.max_input_length)
        max_context_length = (
            int(context_limit)
            if context_limit is not None
            else max_input_length + int(self.trajectory_runner_cfg.sampling_params.max_generate_length)
        )

        trajectory_ids = request.get("trajectory_ids")

        async def run(index, task):
            harbor = self.harbor if "harbor" in task.metadata else None
            inference_loop = asyncio.get_running_loop()
            session_id = trajectory_ids[index].to_string() if trajectory_ids is not None else uuid4().hex

            async def model(request):
                pending = asyncio.run_coroutine_threadsafe(
                    _model_turn(
                        self.model_client,
                        request,
                        sampling_params=sampling,
                        max_context_length=max_context_length,
                        max_prompt_length=max_input_length if context_limit is None else None,
                        chat_template_kwargs=dict(self.trajectory_runner_cfg.get("chat_template_kwargs", {})),
                        session_id=session_id,
                    ),
                    inference_loop,
                )
                return await asyncio.wrap_future(pending)

            engine = ShellboxRolloutEngine(
                model,
                self.factories,
                max_turns=(
                    harbor.max_turns
                    if harbor is not None and harbor.max_turns is not None
                    else self.trajectory_runner_cfg.max_turns
                ),
                command_timeout=self.command_timeout,
                convention=SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
                sessions={
                    GYM_INTERACTION: partial(
                        GymTaskSession,
                        max_turns=self.trajectory_runner_cfg.max_turns,
                        executor=self.environment_executor,
                    )
                },
            )
            harbor_slots = (
                nullcontext()
                if harbor is None
                else self.harbor_eval_slots
                if phase == "eval"
                else self.harbor_task_slots
            )
            retries = 0
            while True:
                try:
                    async with harbor_slots, self.task_slots:
                        result = await _run_engine_task(engine, task, self.engine_executor)
                except RolloutInterrupted as interruption:
                    result = _failed_rollout(
                        interruption,
                        task,
                        self.error_handling if harbor is None else harbor.error_handling,
                        logprobs_required=sampling.get("logprobs") is not None,
                    )
                if harbor is None:
                    return result
                result = harbor_grading_failure(result)
                delay = harbor.retry_delay(result.failure, retries)
                if delay is None:
                    return replace(result, metrics={**result.metrics, "rollout_retries": float(retries)})
                logger.info("Task {} retry {} after {} seconds", task.id, retries + 1, delay)
                with rollout_wait("retry_backoff"):
                    await self.retry_wait(delay)
                retries += 1

        with rollout_phase("collect"):
            async with asyncio.TaskGroup() as group:
                pending = [group.create_task(run(index, task)) for index, task in enumerate(tasks)]
            rollouts = await asyncio.to_thread(
                grade_groups,
                tasks,
                [task.result() for task in pending],
                [item.instance_id for item in trajectory_ids]
                if trajectory_ids is not None
                else [task.id for task in tasks],
                self.group_graders,
                phase,
                [
                    self.harbor.error_handling
                    if self.harbor is not None and "harbor" in task.metadata
                    else self.error_handling
                    for task in tasks
                ],
                logprobs_required=sampling.get("logprobs") is not None,
            )
            if self.harbor is not None:
                rollouts = shape_harbor_rollouts(
                    rollouts,
                    request,
                    self.harbor,
                    int(self.trajectory_runner_cfg.sampling_params.max_generate_length),
                    self.projection.projection.tokenizer,
                    harbor_tasks=["harbor" in task.metadata for task in tasks],
                )
        return rollouts

    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        outputs = await self.generate(input_batch)
        return await self.training_batch(input_batch, outputs)

    async def training_batch(self, request: TrajectoryRequestBatch, outputs: Sequence[RolloutData]) -> TrajectoryBatch:
        with rollout_phase("assemble"):
            batch = self.projection.project(outputs, request)
            propagate_data_sources(request, batch)
        return await finalize_trajectory_batch(request, batch, self.trajectory_runner_cfg, self.trajectory_sink)

    def set_trajectory_sink(self, sink: RetentionSink) -> None:
        sink.bind_runner(type(self).__name__)
        self.trajectory_sink = sink

    async def startup(self) -> None:
        """Resources are scoped to each task and require no worker initialization."""

    async def shutdown(self) -> None:
        """Release the worker threads after all task sessions close."""
        await asyncio.to_thread(self.engine_executor.shutdown, wait=True)
        if self.environment_executor is not None:
            await asyncio.to_thread(self.environment_executor.shutdown, wait=True)

    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None) -> None:
        """Each evaluation task has its own session."""

    async def stop_eval_session(self) -> None:
        """Evaluation resources close with their tasks."""

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        rollouts = await self.generate(task.request)
        batch = await self.training_batch(task.request, rollouts)
        group = RolloutGroup(batch, task.prompt["uid"], task.lease.policy_step, task.prompt)
        with rollout_wait("enqueue"):
            await writer.write_rollout(task.lease, group)
        return sum(len(response) for response in batch["response_ids"])


@dataclass(frozen=True)
class TaskRolloutWorkerSpec:
    config: DictConfig
    engines: list[InferenceEngineInterface]
    environments: dict[str, EnvSpec]
    harbor_config: DictConfig | None = None

    @classmethod
    def from_config(cls, config, engines, *, harbor_config: DictConfig | None = None):
        return cls(
            detached_config(config),
            list(engines),
            dict(registry),
            None if harbor_config is None else detached_config(harbor_config),
        )

    def build(self, tokenizer, shard: WorkerShard) -> TaskRolloutWorker:
        registry.update(self.environments)
        harbor = None if self.harbor_config is None else HarborTaskSettings.from_config(self.harbor_config)
        runner_config = self.config.generator
        client_config = OmegaConf.merge(self.config, {"generator": {"enable_http_endpoint": False}})
        client = DirectModelClient(InferenceEngineClient(self.engines, tokenizer, client_config))
        return TaskRolloutWorker(
            runner_config,
            (
                StepTaskProjection(StepWiseTrajectoryProjection(runner_config, tokenizer))
                if self.config.trainer.step_wise_training
                else WholeTaskProjection(WholeTrajectoryProjection(runner_config, tokenizer))
            ),
            client,
            {
                EnvironmentKind.DOCKER: harbor.machine_factory(self.config.trajectory_runner)
                if harbor is not None
                else DockerMachineFactory(
                    skopeo=Path(self.config.trajectory_runner.skopeo),
                    image_cache=Path(self.config.trajectory_runner.image_cache).expanduser(),
                ),
                EnvironmentKind.SHELLSIM: ShellSimMachineFactory(),
            },
            command_timeout=float(self.config.trajectory_runner.command_timeout),
            max_env_workers=int(self.config.environment.skyrl_gym.max_env_workers),
            harbor=harbor,
            concurrent_tasks=(
                self.config.trajectory_runner.rollout_workers.executor_threads
                if self.config.trajectory_runner.max_concurrent_tasks is None
                else self.config.trajectory_runner.max_concurrent_tasks
            ),
            concurrent_harbor_tasks=None if harbor is None else max(1, harbor.concurrent_trials // shard.count),
        )
