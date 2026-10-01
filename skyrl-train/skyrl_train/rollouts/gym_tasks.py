"""Task conversion and interaction adapters for the existing Gym graders."""

import asyncio
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any

import skyrl_gym
from omegaconf import OmegaConf
from skyrl_gym.verification import VerificationStatus
from taskcompendium.environment import ExternalVerifierSpec
from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.models import TaskSpec, VerifierKind
from rolloutengine.contracts import ModelTurn, RolloutContractError, Transition
from taskcompendium.submission import conversation_messages

from skyrl_train.rollout_observability import run_environment
from skyrl_train.trajectory_runners.skyrl_gym_contracts import (
    environment_metrics_from_step,
    fold_verification_results,
    publish_rollout_evidence,
    reward_from_env_step,
    verification_from_env_step,
)


async def _environment_operation[T](executor: Executor | None, operation: Callable[..., T], *args: Any) -> T:
    """Let a started environment operation finish before cancellation closes its resources."""
    pending = asyncio.create_task(run_environment(executor, operation, *args))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                continue
        pending.result()
        raise


class GymTaskSession:
    """Run environment operations without a model client or an inference loop."""

    def __init__(self, task: TaskSpec, max_turns: int, executor: Executor | None = None):
        if task.verifier.kind != VerifierKind.EXTERNAL:
            raise ValueError("Gym tasks require a private external verifier")
        self.task = task
        specification = ExternalVerifierSpec.model_validate_json(task.verifier.parameters_json)
        self.name = specification.name
        self.parameters = specification.parameters
        self.max_turns = max_turns
        self.executor = executor
        self.environment = None
        self.verifications = []

    async def prepare(self) -> dict[str, Any]:
        messages, metadata = await _environment_operation(self.executor, self._prepare)
        options = dict(metadata.get("chat_completion_params") or {})
        options.pop("input", None)
        if options.get("tools"):
            options["tools"] = [
                {"type": "function", "function": {key: value for key, value in tool.items() if key != "type"}}
                for tool in options["tools"]
            ]
        return {"messages": messages, **options}

    def _prepare(self):
        extras = {**self.parameters["extras"], "max_turns": self.max_turns}
        self.environment = skyrl_gym.make(
            self.name,
            env_config=OmegaConf.create(self.parameters["config"]),
            extras=extras,
        )
        return self.environment.init(conversation_messages(self.task.context))

    async def advance(self, turn: ModelTurn) -> Transition:
        assert self.environment is not None
        metadata = {"assistant_message": turn.message}
        budget = turn.metadata.get("generation_token_budget")
        if budget is not None:
            metadata["generation_token_budget"] = budget
        evidence = publish_rollout_evidence(
            self.environment,
            messages=(turn.message,),
            response=turn.text,
            stop_reason=turn.stop_reason,
            prompt_token_ids=turn.prompt_token_ids,
            response_token_ids=turn.response_token_ids,
            behavior_logprobs=turn.logprobs,
            metadata=metadata,
        )
        output = await _environment_operation(self.executor, self.environment.step, turn.text)
        action = output.get("postprocessed_action")
        if action is not None and action != turn.text:
            raise RolloutContractError("An environment cannot replace the sampled model action")
        verification = verification_from_env_step(output)
        reward = reward_from_env_step(output, verification)
        reward.validate_for(evidence)
        self.verifications.append(verification)
        reset = output.get("reset_conversation")
        if reset is not None:
            self.verifications.clear()
        return Transition(
            done=output["done"],
            observations=tuple(output["observations"]),
            reset_conversation=None if reset is None else tuple(reset),
            reward=reward.optimization_reward,
            token_rewards=reward.token_rewards,
            token_credit=reward.token_credit,
            reward_components=dict(reward.components),
            grade=grade_result(verification),
            metrics=environment_metrics_from_step(output, self.environment.get_metrics()),
        )

    async def grade(self, messages) -> GradeResult:
        del messages
        verification, _ = fold_verification_results(self.verifications)
        return grade_result(verification)

    async def close(self) -> None:
        if self.environment is not None:
            await _environment_operation(self.executor, self.environment.close)


def grade_result(verification) -> GradeResult:
    if verification.status == VerificationStatus.VERIFIED:
        return GradeResult(
            Outcome.GRADED,
            verification.score,
            passed=verification.passed,
            diagnostics=dict(verification.diagnostics),
            score_min=verification.score_min,
            score_max=verification.score_max,
        )
    statuses = {
        VerificationStatus.SKIPPED: Outcome.SKIPPED,
        VerificationStatus.UNAVAILABLE: Outcome.UNAVAILABLE,
        VerificationStatus.ERROR: Outcome.INFRA_ERROR,
    }
    return GradeResult(
        statuses[verification.status], None, verification.reason, diagnostics=dict(verification.diagnostics)
    )
