import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from rolloutengine.contracts import ModelTurn, RolloutData, SessionStart
from taskcompendium.grading_result import Outcome
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from skyrl_train.rollouts.agent_backend import AgentInferenceBackend, TRIAL_HEADER
from skyrl_train.rollouts.harbor_settings import HARBOR_EXCEPTION
from skyrl_train.rollouts.harbor_tasks import harbor_grading_failure
from skyrl_train.rollouts.harbor_trial_session import HarborTrialSession, run_harbor_trial, trial_grade
from skyrl_train.rollouts.task_projections import rollout_loss_eligible
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig


@pytest.mark.asyncio
async def test_harbor_trial_delivers_completion_and_retains_numeric_router_evidence(tmp_path):
    class Agent(BaseInstalledAgent):
        rollout_correlation_id = "trial"

        @staticmethod
        def name():
            return "fixture"

        async def install(self, environment):
            raise AssertionError("No remote installation in this test")

        async def run(self, instruction, environment, context):
            raise AssertionError("The trial fixture drives model requests directly")

    async def tokenize(payload):
        return {"tokens": [2], "count": 1, "max_model_len": 32}

    routes = np.array([[[3, 7]]], dtype=np.int32)
    turn = ModelTurn(
        {"role": "assistant", "content": "done"},
        (2,),
        (1,),
        (-0.25,),
        "stop",
        metadata={"routed_experts": routes, "count": np.int64(1)},
    )

    async def complete(request):
        return turn

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0})))
    backend = AgentInferenceBackend("model", 32, tokenize, tokenizer)
    result = completed_trial(1.0)

    async def run():
        response = await backend.chat_completion(
            {
                "headers": {TRIAL_HEADER: "trial"},
                "json": {"model": "model", "messages": [{"role": "user", "content": "Task"}]},
            }
        )
        assert response["choices"][0]["message"]["content"] == "done"
        return result

    # Replace the remote trial boundary; exercise the real request recorder and backend.
    trial = SimpleNamespace(agent=Agent(logs_dir=tmp_path / "agent", version="fixture"), run=run)
    assert await run_harbor_trial(trial, backend, complete, tmp_path / "engine") is result
    records = [json.loads(line) for line in (tmp_path / "engine" / "turns.jsonl").read_text().splitlines()]
    assert records[0]["request"]["messages"] == [{"role": "user", "content": "Task"}]
    assert records[1]["turn"]["metadata"] == {"routed_experts": [[[3, 7]]], "count": 1}
    assert turn.metadata["routed_experts"] is routes


def completed_trial(score, exception_type=None):
    task = TaskConfig(path="original-task")
    return TrialResult(
        task_name="original-task",
        trial_name="attempt",
        trial_uri="/tmp/attempt",
        task_id=task.get_task_id(),
        task_checksum="checksum",
        config=TrialConfig(task=task),
        agent_info=AgentInfo(name="mini-swe-agent", version="2.1.0"),
        verifier_result=None if score is None else VerifierResult(rewards={"reward": score}),
        exception_info=None
        if exception_type is None
        else ExceptionInfo(
            exception_type=exception_type,
            exception_message="phase ended",
            exception_traceback="",
            occurred_at=datetime.now(timezone.utc),
        ),
    )


@pytest.mark.parametrize("score", [0.0, 1.0, None])
def test_agent_timeout_keeps_available_grade_without_inventing_zero(score):
    grade = trial_grade(completed_trial(score, "AgentTimeoutError"), "reward")
    assert grade.reward == score
    assert grade.status == (Outcome.INFRA_ERROR if score is None else Outcome.GRADED)
    assert grade.diagnostics[HARBOR_EXCEPTION] == "AgentTimeoutError"


@pytest.mark.parametrize(
    "exception_type,score,preserve_timeout,eligible",
    [
        ("AgentTimeoutError", 1.0, True, True),
        ("AgentTimeoutError", 0.0, True, True),
        ("AgentTimeoutError", None, True, False),
        ("AgentTimeoutError", 1.0, False, False),
        ("DaytonaConnectionError", 1.0, True, False),
    ],
)
def test_trial_failure_policy_preserves_grades_and_masks_infrastructure(
    exception_type, score, preserve_timeout, eligible
):
    rollout = harbor_grading_failure(
        RolloutData(
            task_id="original-task",
            messages=({"role": "assistant", "content": "edit"},),
            prompt_token_ids=(1,),
            response_token_ids=(2,),
            loss_mask=(1,),
            logprobs=(-0.3,),
            grade=trial_grade(completed_trial(score, exception_type), "reward"),
            stop_reason="stop",
        )
    )
    policy = ErrorHandlingConfig(
        enable_error_classification=True,
        passthrough_exceptions=frozenset({"AgentTimeoutError"}),
        mask_exceptions=frozenset({"DaytonaConnectionError"}),
        preserve_logprobs_on_timeout=preserve_timeout,
    )
    assert rollout.grade.reward == score
    assert rollout.failure.exception_type == exception_type
    assert rollout_loss_eligible(rollout, policy, logprobs_required=True) == eligible


@pytest.mark.asyncio
async def test_trial_session_preserves_turns_and_returns_original_verifier_score():
    delivered = []

    async def run(complete):
        history = ({"role": "user", "content": "Repair the repository."},)
        first = await complete(SessionStart(history, {}))
        delivered.append(first)
        second = await complete(
            SessionStart((*history, first.message, {"role": "user", "content": "Tests passed."}), {})
        )
        delivered.append(second)
        return completed_trial(1.0)

    session = HarborTrialSession(run, "reward")
    first = ModelTurn({"role": "assistant", "content": "edit"}, (11,), (12,), (-0.3,), "stop")
    second = ModelTurn({"role": "assistant", "content": "done"}, (11, 12, 13), (14,), (-0.2,), "stop")
    try:
        request = await session.prepare()
        assert request.messages[0]["content"] == "Repair the repository."
        transition = await session.advance(first)
        assert not transition.done
        assert transition.observations == ({"role": "user", "content": "Tests passed."},)
        assert (await session.advance(second)).done
        grade = await session.grade((), stop_reason="stop", outputs={})
        assert grade.status == Outcome.GRADED and grade.reward == 1.0
        assert delivered == [first, second]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_trial_session_cancellation_finishes_trial_cleanup():
    cleaned = asyncio.Event()

    async def run(complete):
        try:
            await complete(SessionStart(({"role": "user", "content": "Task"},), {}))
            raise AssertionError("No model response should arrive")
        finally:
            cleaned.set()

    session = HarborTrialSession(run, "reward")
    await session.prepare()
    await session.close()
    assert cleaned.is_set()
    grade = await session.grade((), stop_reason="cancelled", outputs={})
    assert grade.status == Outcome.UNAVAILABLE and grade.reward is None
