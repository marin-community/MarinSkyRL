"""TaskCompendium execution and projection into SkyRL training batches."""

import asyncio
import shutil
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from jinja2 import TemplateError
from harbor_config.models.environment_type import EnvironmentType
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Machine, MachineFactory
from taskcompendium.grading_result import Outcome
from rolloutengine.contracts import (
    GenerationLimitReached,
    ModelRequest,
    ModelTurn,
    RolloutContractError,
    RolloutData,
    RolloutFailure,
    RolloutInterrupted,
    RolloutOperation,
    TaskSession,
)
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.lowering import SHELLBOX_SESSION, validate_lowered_task
from skyrl_train.dataset.tasks import LOWERED_TASK_COLUMN
from skyrl_train.config.utils import generation_context_limit
from taskcompendium.submission import PlainText
from skyrl_gym.task_factories import session_factories
from skyrl_gym.task_records import fold_grades
from skyrl_gym.verification import VERIFIER_RUNTIME_ERROR

from skyrl_train.inference_engines.base import ChatContinuation, InferenceEngineInterface, InferenceEngineInput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.rollouts.buffer import RolloutGroup, RolloutTask, RolloutWriter
from skyrl_train.rollouts.group_grader import GroupGraderSpec
from skyrl_train.rollouts.group_grading import GROUP_GRADERS, GroupGrader, grade_groups
from skyrl_train.rollouts.harbor_tasks import (
    HARBOR_MACHINE_BACKEND,
    HarborTaskSettings,
    harbor_grading_failure,
    shape_harbor_rollouts,
)
from skyrl_train.rollouts.machines import OwnedMachineFactory, TaskMachineError
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
    logprobs_requested,
)
from skyrl_train.rollouts.task_projections import (
    StepTaskProjection,
    WholeTaskProjection,
)
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
    lowered: LoweredTaskSpec,
) -> RolloutData:
    """Retain verified turns and safe diagnostics for the training policy."""
    task = lowered.task
    error = interruption.__cause__
    assert isinstance(error, Exception)
    if interruption.operation == RolloutOperation.MODEL and not isinstance(
        error, (ModelServerError, TimeoutError, ConnectionError, TemplateError)
    ):
        raise error
    exception_type = type(error).__name__
    diagnostics = {"operation": interruption.operation.value, "timed_out": isinstance(error, TimeoutError)}
    if isinstance(error, TaskMachineError) and error.__cause__ is not None:
        diagnostics["cause_error_type"] = type(error.__cause__).__name__
    if isinstance(error, TimeoutError):
        exception_type = {
            RolloutOperation.ATTEMPT: "TrialTimeoutError",
            RolloutOperation.START: "EnvironmentStartTimeoutError",
            RolloutOperation.PREPARE: "AgentSetupTimeoutError",
            RolloutOperation.GRADE: "VerifierTimeoutError",
            RolloutOperation.CLEANUP: "CleanupTimeoutError",
        }.get(interruption.operation, "AgentTimeoutError")
    elif isinstance(error, TemplateError):
        exception_type = "TemplateError"
        diagnostics["cause_error_type"] = type(error).__name__
    elif isinstance(error, TaskMachineError):
        exception_type = "TaskMachineError"
    elif interruption.operation in {RolloutOperation.START, RolloutOperation.PREPARE}:
        exception_type = VERIFIER_RUNTIME_ERROR
        diagnostics.setdefault("cause_error_type", type(error).__name__)
    elif isinstance(error, ModelServerError):
        if error.category == "context_overflow":
            exception_type = "ContextLengthExceededError"
        diagnostics.update(error_category=error.category, request_id=error.request_id, status_code=error.status_code)
    rollout = interruption.rollout
    if (
        interruption.operation == RolloutOperation.ADVANCE
        and lowered.session.task_session != SHELLBOX_SESSION
        and rollout.steps
        and rollout.steps[-1].transition.metrics.get("advance_incomplete")
    ):
        # A native grader failure cannot supply credit for its incomplete turn.
        steps = rollout.steps[:-1]
        end = steps[-1].response_end + 1 if steps else 0
        rollout = replace(
            rollout,
            response_token_ids=rollout.response_token_ids[:end],
            loss_mask=rollout.loss_mask[:end],
            logprobs=None if rollout.logprobs is None else rollout.logprobs[:end],
            messages=steps[-1].messages if steps else (),
            steps=steps,
        )
    grade = rollout.grade
    if grade.status == Outcome.UNAVAILABLE and interruption.operation != RolloutOperation.GRADE and rollout.steps:
        grade = fold_grades([step.transition.grade for step in rollout.steps if step.transition.grade is not None])
    if cleanup_errors := interruption.rollout.grade.diagnostics.get("cleanup_errors"):
        grade = replace(grade, diagnostics={**grade.diagnostics, "cleanup_errors": cleanup_errors})
    logger.warning("Task {} interrupted during {}: {}", task.id, interruption.operation, exception_type)
    return replace(
        rollout,
        grade=grade,
        stop_reason="error",
        failure=RolloutFailure(exception_type, diagnostics),
    )


class TaskRolloutWorker:
    """Run canonical task rollouts and submit complete groups to the buffer."""

    def __init__(
        self,
        trajectory_runner_cfg: DictConfig,
        projection: WholeTaskProjection | StepTaskProjection,
        model_client: ModelClient,
        factories: Mapping[str, MachineFactory],
        max_verifier_workers: int = 0,
        harbor: HarborTaskSettings | None = None,
        concurrent_tasks: int | None = None,
        concurrent_harbor_tasks: int | None = None,
        retry_wait: Callable[[float], Awaitable[None]] = asyncio.sleep,
        group_graders: Mapping[str, GroupGrader] = GROUP_GRADERS,
        sessions: Mapping[str, Callable[[LoweredTaskSpec, Machine | None], TaskSession]] | None = None,
        *,
        shutdown_timeout: float,
    ):
        self.trajectory_runner_cfg = trajectory_runner_cfg
        self.model_client = model_client
        self.factories = {name: OwnedMachineFactory(factory, name) for name, factory in factories.items()}
        self.active_requests: set[asyncio.Task] = set()
        self.closing = False
        self.shutdown_timeout = shutdown_timeout
        self.chat_template_kwargs = dict(trajectory_runner_cfg.get("chat_template_kwargs", {}))
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
        self.verifier_executor = (
            ThreadPoolExecutor(max_workers=max_verifier_workers, thread_name_prefix="task-verifier")
            if max_verifier_workers > 0
            else None
        )
        self.sessions = session_factories(executor=self.verifier_executor)
        if sessions is not None:
            self.sessions.update(sessions)
        self.trajectory_sink: RetentionSink | None = None

    @contextmanager
    def _active_request(self):
        if self.closing:
            raise RuntimeError("The rollout worker is closed")
        task = asyncio.current_task()
        assert task is not None
        if task in self.active_requests:
            yield
            return
        self.active_requests.add(task)
        try:
            yield
        finally:
            self.active_requests.discard(task)

    async def generate(self, request: TrajectoryRequestBatch) -> list[RolloutData]:
        with self._active_request():
            return await self._generate(request)

    async def _generate(self, request: TrajectoryRequestBatch) -> list[RolloutData]:
        extras = request.get("env_extras")
        if extras is None or len(extras) != len(request["prompts"]):
            raise ValueError("Each rollout request must contain one lowered task per prompt")
        lowered_tasks = [LoweredTaskSpec.model_validate_json(row[LOWERED_TASK_COLUMN]) for row in extras]
        group_graders = [
            GroupGraderSpec.model_validate_json(row["group_grader"]) if row.get("group_grader") is not None else None
            for row in extras
        ]
        metadata = request.get("batch_metadata")
        phase = "train" if metadata is None else metadata.training_phase
        harbor_settings = [self.harbor if "harbor" in item.task.tags else None for item in lowered_tasks]
        policies = [self.error_handling if harbor is None else harbor.error_handling for harbor in harbor_settings]
        lowered_tasks = [
            item if harbor is None else harbor.lowered(item, phase=phase)
            for item, harbor in zip(lowered_tasks, harbor_settings, strict=True)
        ]
        for lowered in lowered_tasks:
            validate_lowered_task(lowered, factories=self.factories, sessions=self.sessions)
        tasks = [item.task for item in lowered_tasks]
        sampling = get_sampling_params_for_backend(
            self.trajectory_runner_cfg.backend, self.trajectory_runner_cfg.sampling_params
        )
        sampling.update(request.get("sampling_params") or {})
        require_logprobs = logprobs_requested(request, self.trajectory_runner_cfg)
        max_input_length = int(self.trajectory_runner_cfg.max_input_length)
        max_context_length = generation_context_limit(self.trajectory_runner_cfg)

        trajectory_ids = request.get("trajectory_ids")

        async def run(index, lowered):
            task = lowered.task
            harbor = harbor_settings[index]
            session_id = trajectory_ids[index].to_string() if trajectory_ids is not None else uuid4().hex

            async def model(request):
                return await _model_turn(
                    self.model_client,
                    request,
                    sampling_params=sampling,
                    max_context_length=max_context_length,
                    max_prompt_length=max_input_length,
                    chat_template_kwargs=self.chat_template_kwargs,
                    session_id=session_id,
                )

            engine = ShellboxRolloutEngine(
                model,
                self.factories,
                convention=PlainText(id="plain"),
                sessions=self.sessions,
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
                        result = await engine.run(lowered)
                except RolloutInterrupted as interruption:
                    result = _failed_rollout(interruption, lowered)
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
                pending = [group.create_task(run(index, item)) for index, item in enumerate(lowered_tasks)]
            rollouts = await grade_groups(
                tasks,
                group_graders,
                [task.result() for task in pending],
                [item.instance_id for item in trajectory_ids]
                if trajectory_ids is not None
                else [task.id for task in tasks],
                self.group_graders,
                phase,
                policies,
                logprobs_required=require_logprobs,
            )
            if self.harbor is not None:
                rollouts = shape_harbor_rollouts(
                    rollouts,
                    request,
                    self.harbor,
                    int(self.trajectory_runner_cfg.sampling_params.max_generate_length),
                    self.projection.projection.tokenizer,
                    harbor_tasks=[harbor is not None for harbor in harbor_settings],
                )
        return rollouts

    async def run(self, input_batch: TrajectoryRequestBatch) -> TrajectoryBatch:
        with self._active_request():
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
        pass

    async def shutdown(self) -> None:
        """Cancel rollouts and release their machines before actor termination."""
        self.closing = True
        deadline = asyncio.get_running_loop().time() + self.shutdown_timeout
        for factory in self.factories.values():
            factory.stop_creating()
        pending = tuple(self.active_requests)
        for task in pending:
            task.cancel()
        failures = []
        try:
            if pending:
                _, remaining = await asyncio.wait(pending, timeout=self.shutdown_timeout / 2)
                if remaining:
                    logger.error("Task worker shutdown deadline expired with {} active rollouts", len(remaining))
                    failures.append(TimeoutError("Task rollouts did not stop before the shutdown deadline"))
            results = await asyncio.gather(
                *(factory.close(deadline) for factory in self.factories.values()), return_exceptions=True
            )
            failures.extend(result for result in results if isinstance(result, Exception))
        finally:
            if self.verifier_executor is not None:
                self.verifier_executor.shutdown(wait=False, cancel_futures=True)
        if failures:
            raise ExceptionGroup("Task worker shutdown failed", failures)

    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None) -> None:
        pass

    async def stop_eval_session(self) -> None:
        pass

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        with self._active_request():
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
    sessions: dict[str, Callable[[LoweredTaskSpec, Machine | None], TaskSession]] = field(default_factory=dict)
    harbor_config: DictConfig | None = None

    @classmethod
    def from_config(
        cls,
        config,
        engines,
        *,
        harbor_config: DictConfig | None = None,
        sessions: Mapping[str, Callable[[LoweredTaskSpec, Machine | None], TaskSession]] | None = None,
    ):
        return cls(
            detached_config(config),
            list(engines),
            {} if sessions is None else dict(sessions),
            None if harbor_config is None else detached_config(harbor_config),
        )

    def build(self, tokenizer, shard: WorkerShard) -> TaskRolloutWorker:
        harbor = None if self.harbor_config is None else HarborTaskSettings.from_config(self.harbor_config)
        runner_config = self.config.generator
        client_config = OmegaConf.merge(self.config, {"generator": {"enable_http_endpoint": False}})
        client = DirectModelClient(InferenceEngineClient(self.engines, tokenizer, client_config))
        factories: dict[str, MachineFactory] = {"shellsim": ShellSimMachineFactory()}
        skopeo = shutil.which(str(Path(self.config.trajectory_runner.skopeo).expanduser()))
        if shutil.which("docker") is not None and skopeo is not None:
            factories["docker"] = DockerMachineFactory(
                skopeo=Path(skopeo),
                image_cache=Path(self.config.trajectory_runner.image_cache).expanduser(),
            )
        if harbor is not None:
            if harbor.environment.type != EnvironmentType.DOCKER:
                factories[HARBOR_MACHINE_BACKEND] = harbor.machine_factory(self.config.trajectory_runner)
            elif "docker" in factories:
                factories[HARBOR_MACHINE_BACKEND] = factories["docker"]
        return TaskRolloutWorker(
            runner_config,
            (
                StepTaskProjection(StepWiseTrajectoryProjection(runner_config, tokenizer))
                if self.config.trainer.step_wise_training
                else WholeTaskProjection(WholeTrajectoryProjection(runner_config, tokenizer))
            ),
            client,
            factories,
            max_verifier_workers=int(self.config.environment.task_sessions.max_verifier_workers),
            harbor=harbor,
            concurrent_tasks=int(self.config.trajectory_runner.max_concurrent_tasks),
            concurrent_harbor_tasks=None if harbor is None else max(1, harbor.concurrent_trials // shard.count),
            sessions=self.sessions,
            shutdown_timeout=float(self.config.trajectory_runner.shutdown_timeout),
        )
