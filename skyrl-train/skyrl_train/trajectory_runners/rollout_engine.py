"""Native TaskCompendium execution through Marin's RolloutEngine."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from uuid import uuid4

import numpy as np
from omegaconf import DictConfig, OmegaConf
from rolloutengine.contracts import (
    ModelRequest,
    ModelTurn,
    RolloutContractError,
    RolloutData,
    RolloutInterrupted,
    RolloutOperation,
)
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.spec import LoweredTaskSpec
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.machine import MachineFactory
from taskcompendium.grading_result import Outcome
from taskcompendium.submission import PlainText, conversation_messages
from transformers import PreTrainedTokenizerBase

from skyrl_gym.verification import RewardResult, RolloutEvidence, TrainingDisposition, VerificationResult
from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.inference_engines.base import ChatContinuation, InferenceEngineInput, InferenceEngineInterface
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.rollouts.workers import WorkerShard, detached_config
from skyrl_train.trajectory_runners.base import TrajectoryRunner, propagate_data_sources
from skyrl_train.trajectory_runners.model_clients import DirectModelClient, ModelClient, ModelServerError
from skyrl_train.trajectory_runners.projections import WholeTrajectoryProjection
from skyrl_train.trajectory_runners.selected_topk import align_student_topk
from skyrl_train.trajectory_runners.types import (
    AgentLoopOutput,
    TokenProvenance,
    TrajectoryBatch,
    TrajectoryRequestBatch,
)
from skyrl_train.utils.harbor_errors import (
    ErrorHandlingConfig,
    classify_exception_type,
    passthrough_logprob_error_type,
    treatment_excludes_from_baseline,
)

LOWERED_TASK_COLUMN = "lowered_task_json"
MASKED_TOKEN_ID = 0
_TIMEOUT_EXCEPTION_TYPES = {
    RolloutOperation.START: "EnvironmentStartTimeoutError",
    RolloutOperation.PREPARE: "AgentSetupTimeoutError",
    RolloutOperation.MODEL: "AgentTimeoutError",
    RolloutOperation.ADVANCE: "AgentTimeoutError",
    RolloutOperation.ATTEMPT: "TrialTimeoutError",
    RolloutOperation.GRADE: "VerifierTimeoutError",
}


def validate_rollout_engine_config(config: DictConfig) -> None:
    if config.trainer.step_wise_training:
        raise NotImplementedError("The RolloutEngine entrypoint supports whole-rollout training")
    if config.data.get("terminal_bench_data"):
        raise NotImplementedError("The RolloutEngine entrypoint accepts lowered task records, not Harbor task roots")


def _training_output(
    rollout: RolloutData,
    interruption: RolloutInterrupted | None,
    error_handling: ErrorHandlingConfig,
    *,
    logprobs_required: bool,
) -> AgentLoopOutput:
    grade = rollout.grade
    graded = grade.status in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}
    if graded:
        assert grade.reward is not None
        verification = VerificationResult.verified(
            grade.reward,
            passed=grade.passed,
            diagnostics=grade.diagnostics,
            score_min=grade.score_min,
            score_max=grade.score_max,
        )
        disposition = TrainingDisposition.train()
    elif grade.status == Outcome.SKIPPED:
        verification = VerificationResult.skipped(grade.error or "Verification skipped")
        disposition = TrainingDisposition.mask("No verifier verdict", exception_type=grade.status.value)
    elif grade.status == Outcome.UNAVAILABLE:
        verification = VerificationResult.unavailable(grade.error or "No verifier verdict")
        disposition = TrainingDisposition.mask("No verifier verdict", exception_type=grade.status.value)
    else:
        verification = VerificationResult.error(grade.error or grade.status.value, diagnostics=grade.diagnostics)
        disposition = TrainingDisposition.mask("No verifier verdict", exception_type=grade.status.value)

    optimization_reward = verification.score if verification.score is not None else 0.0
    error_treatment = None
    if interruption is not None:
        error = interruption.__cause__
        assert isinstance(error, Exception)
        exception_type = type(error).__name__
        diagnostics = {"error_type": exception_type, "operation": interruption.operation.value}
        if isinstance(error, ModelServerError):
            if error.category == "context_overflow":
                exception_type = "ContextLengthExceededError"
            diagnostics.update(
                error_category=error.category, request_id=error.request_id, status_code=error.status_code
            )
        elif isinstance(error, TimeoutError):
            exception_type = _TIMEOUT_EXCEPTION_TYPES.get(interruption.operation, exception_type)
        diagnostics["exception_type"] = exception_type
        diagnostics = {**verification.diagnostics, **diagnostics}
        verification = (
            replace(verification, diagnostics=diagnostics)
            if graded
            else VerificationResult.error(str(error), diagnostics=diagnostics)
        )
        treatment = (
            classify_exception_type(exception_type, error_handling)
            if error_handling.enable_error_classification
            else ErrorTreatment.MASK
        )
        missing_logprobs = passthrough_logprob_error_type(
            treatment,
            has_rollout_logprobs=rollout.logprobs is not None,
            rollout_logprobs_required=logprobs_required,
        )
        baseline_eligible = (
            not treatment_excludes_from_baseline(treatment, verifier_available=graded) and missing_logprobs is None
        )
        disposition = TrainingDisposition(
            loss_eligible=any(rollout.loss_mask) and baseline_eligible,
            baseline_eligible=baseline_eligible,
            reason="Rollout execution failed",
            exception_type=missing_logprobs or exception_type,
        )
        error_treatment = treatment.value
        if treatment is not ErrorTreatment.PASSTHROUGH or not graded:
            optimization_reward = 0.0

    candidates = [step.turn.metadata.get("student_topk_indices") for step in rollout.steps]
    scores = [step.turn.metadata.get("behavior_topk_logprobs") for step in rollout.steps]
    selected = None
    if any(value is not None for value in candidates):
        if any(value is None for value in (*candidates, *scores)):
            raise RolloutContractError("Student top-K evidence must be present on every model turn")
        selected = align_student_topk(
            rollout.response_token_ids,
            rollout.loss_mask,
            [token for step in rollout.steps for token in step.turn.response_token_ids],
            [row for turn in candidates for row in turn],
            [row for turn in scores for row in turn],
        )
    routes = [step.turn.metadata.get("routed_experts") for step in rollout.steps]
    template = next((value for value in routes if value is not None), None)
    routed_experts = None
    if template is not None:
        routed_experts = np.zeros((len(rollout.response_token_ids), *template.shape[1:]), dtype=template.dtype)
        for step, values in zip(rollout.steps, routes, strict=True):
            if values is not None:
                start = step.response_end + 1 - len(step.turn.response_token_ids)
                routed_experts[start : step.response_end + 1] = values
    response_token_ids = rollout.response_token_ids
    loss_mask = rollout.loss_mask
    logprobs = rollout.logprobs
    if not response_token_ids:
        # The trainer requires a nonempty row. This token is not sampled evidence.
        response_token_ids, loss_mask, logprobs = (MASKED_TOKEN_ID,), (0,), (0.0,)
    if logprobs is None and not disposition.loss_eligible:
        logprobs = (0.0,) * len(response_token_ids)
    return AgentLoopOutput(
        evidence=RolloutEvidence(
            messages=rollout.messages,
            response=rollout.steps[-1].turn.text if rollout.steps else None,
            stop_reason=rollout.stop_reason,
            generated_token_count=sum(rollout.loss_mask),
            prompt_token_ids=rollout.prompt_token_ids or (MASKED_TOKEN_ID,),
            response_token_ids=response_token_ids,
            behavior_logprobs=None if logprobs is None else np.asarray(logprobs, dtype=np.float32),
            student_topk_indices=None if selected is None else selected.indices,
            behavior_topk_logprobs=None if selected is None else selected.topk_logprobs,
            routed_experts=routed_experts,
        ),
        verification=verification,
        reward=RewardResult(
            unshaped_reward=verification.score,
            optimization_reward=optimization_reward,
        ),
        disposition=disposition,
        loss_mask=list(loss_mask),
        env_metrics=dict(rollout.metrics),
        error_treatment=error_treatment,
    )


class RolloutEngineTrajectoryRunner(TrajectoryRunner):
    """Run native task records and return whole-rollout SkyRL training samples."""

    def __init__(
        self,
        generator: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        model_client: ModelClient,
        factories: Mapping[str, MachineFactory],
    ):
        self.trajectory_runner_cfg = generator
        self.model_client = model_client
        self.factories = factories
        self.projection = WholeTrajectoryProjection(generator, tokenizer)
        self.chat_template_kwargs = dict(generator.get("chat_template_kwargs", {}))
        self.max_context_length = int(generator.max_input_length) + int(generator.sampling_params.max_generate_length)
        self.error_handling = ErrorHandlingConfig.from_mapping(generator.get("error_handling", {}))

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        del disable_tqdm
        extras = input_batch["env_extras"]
        if extras is None:
            raise ValueError(f"RolloutEngine rows require {LOWERED_TASK_COLUMN}")
        tasks = [LoweredTaskSpec.model_validate_json(row[LOWERED_TASK_COLUMN]) for row in extras]
        for task, prompt in zip(tasks, input_batch["prompts"], strict=True):
            if conversation_messages(task.task.context) != prompt:
                raise ValueError("The dataset prompt must match the task context")
        sampling_params = input_batch.get("sampling_params")
        if sampling_params is None:
            sampling_params = get_sampling_params_for_backend(
                self.trajectory_runner_cfg.backend,
                OmegaConf.to_container(self.trajectory_runner_cfg.sampling_params, resolve=True),
            )
        outputs = await asyncio.gather(*(self._run_task(task, sampling_params) for task in tasks))
        output = self.projection.project(outputs, input_batch)
        propagate_data_sources(input_batch, output)
        return output

    async def _run_task(self, task: LoweredTaskSpec, sampling_params: dict) -> AgentLoopOutput:
        session_id = uuid4().hex

        async def model(request: ModelRequest) -> ModelTurn:
            options = dict(request.options)
            # DirectModelClient accepts flat function definitions at its input boundary.
            if "tools" in options:
                options["tools"] = [{"type": "function", **tool["function"]} for tool in options["tools"]]
            if self.chat_template_kwargs:
                options["chat_template_kwargs"] = self.chat_template_kwargs
            continuation = (
                None
                if request.assistant_message_index is None
                else ChatContinuation(
                    served_prefix_token_ids=list(request.prefix_token_ids),
                    assistant_message_index=request.assistant_message_index,
                )
            )
            output = await self.model_client.generate(
                InferenceEngineInput(
                    prompts=[list(request.messages)],
                    chat_completion_params=[options],
                    chat_continuations=[continuation],
                    session_ids=[session_id],
                    sampling_params=sampling_params,
                    max_context_length=self.max_context_length,
                )
            )
            if output["token_provenance"] != TokenProvenance.ENGINE:
                raise RolloutContractError("RolloutEngine requires exact served tokens")
            logprobs = output.get("response_logprobs")
            if sampling_params.get("logprobs") is not None and logprobs is None:
                raise RolloutContractError("The model response omitted requested behavior logprobs")
            return ModelTurn(
                message=output["assistant_messages"][0],
                prompt_token_ids=tuple(output["prompt_ids"][0]),
                response_token_ids=tuple(output["response_ids"][0]),
                logprobs=None if logprobs is None else tuple(logprobs[0]),
                stop_reason=output["stop_reasons"][0],
                text=output["responses"][0],
                metadata={
                    name: output[name][0]
                    for name in ("student_topk_indices", "behavior_topk_logprobs", "routed_experts")
                    if output.get(name) is not None
                },
            )

        engine = ShellboxRolloutEngine(model, self.factories, convention=PlainText(id="plain-text"))
        try:
            rollout = await engine.run(task)
        except RolloutInterrupted as interruption:
            return _training_output(
                interruption.rollout,
                interruption,
                self.error_handling,
                logprobs_required=sampling_params.get("logprobs") is not None,
            )
        return _training_output(
            rollout, None, self.error_handling, logprobs_required=sampling_params.get("logprobs") is not None
        )


@dataclass(frozen=True)
class RolloutEngineRunnerSpec:
    """Serializable inputs for a RolloutEngine runner inside a rollout worker."""

    config: DictConfig
    engines: list[InferenceEngineInterface]

    @classmethod
    def from_config(cls, config: DictConfig, engines: Sequence[InferenceEngineInterface]) -> "RolloutEngineRunnerSpec":
        return cls(detached_config(config), list(engines))

    def build(self, tokenizer: PreTrainedTokenizerBase, shard: WorkerShard) -> TrajectoryRunner:
        del shard
        client_config = OmegaConf.merge(self.config, {"generator": {"enable_http_endpoint": False}})
        client = InferenceEngineClient(self.engines, tokenizer, client_config)
        return RolloutEngineTrajectoryRunner(
            self.config.generator, tokenizer, DirectModelClient(client), {"docker": DockerMachineFactory()}
        )
