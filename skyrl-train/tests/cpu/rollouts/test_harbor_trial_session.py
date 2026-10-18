import asyncio
import json
from datetime import datetime, timezone
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import httpx
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
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.rollouts.harbor_settings import HARBOR_EXCEPTION
from skyrl_train.rollouts.harbor_tasks import harbor_grading_failure
from skyrl_train.rollouts.harbor_trial_session import (
    HarborTrialSession,
    run_harbor_trial,
    trial_grade,
)
from skyrl_train.rollouts.task_projections import rollout_loss_eligible
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.buffer import RolloutLease, RolloutTask
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig, HarborAgentStartupError
from tests.cpu.rollouts.engine_fakes import ConversationClient, Tokenizer as FixtureTokenizer, Writer
from tests.cpu.task_specs import lowered_task, task_input


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
@pytest.mark.parametrize("score", [None, 0.0, 1.0])
async def test_trial_without_inference_preserves_startup_failure_instead_of_a_grade(score):
    async def run(complete):
        result = completed_trial(score, "NonZeroAgentExitCodeError")
        result.exception_info.exception_message = "private endpoint credential"
        return result

    session = HarborTrialSession(run, "reward")
    try:
        with pytest.raises(HarborAgentStartupError) as raised:
            await session.prepare()
        assert raised.value.exception_type == "NonZeroAgentExitCodeError"
        assert raised.value.trial_uri == "/tmp/attempt"
        assert "private" not in str(raised.value)
    finally:
        with pytest.raises(HarborAgentStartupError):
            await session.close()


@pytest.mark.asyncio
async def test_harbor_startup_failure_is_ungraded_and_excluded_from_training(task_inputs):
    config, request = task_inputs
    task = request.rows[0].lowered.task

    async def run(complete):
        return completed_trial(0.0, "NonZeroAgentExitCodeError")

    request = replace(request, rows=(task_input(lowered_task(task, "startup_failure")),))
    client = ConversationClient([])
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(config, FixtureTokenizer()),
        client,
        {},
        sessions={"startup_failure": lambda task, machine: HarborTrialSession(run, "reward")},
        shutdown_timeout=30,
    )
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert client.requests == []
    assert batch["exception_types"] == ["NonZeroAgentExitCodeError"]
    assert batch["exclude_from_baseline"] == [True]
    assert batch["loss_masks"] == [[0]]
    verification = batch["verification_results"][0]
    assert verification.score is None
    assert verification.diagnostics["harbor_trial_uri"] == "/tmp/attempt"


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


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_trial_session_context_limit_releases_request_and_preserves_verification(stream):
    async def tokenize(_payload):
        return {"tokens": [2], "count": 1, "max_model_len": 32}

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0})))
    backend = AgentInferenceBackend("model", 32, tokenize, tokenizer)
    responses = []
    set_global_state(backend, None)

    async def run(complete):
        with backend.session("trial", complete):
            payload = {
                "headers": {TRIAL_HEADER: "trial"},
                "json": {"model": "model", "messages": [{"role": "user", "content": "Task"}]},
            }
            await backend.chat_completion(payload)
            # A real installed agent receives the failed API request, exits, and
            # lets Harbor verify the filesystem. A retry must also terminate.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
            ) as client:
                for _ in range(2):
                    response = await client.post(
                        "/v1/chat/completions", headers=payload["headers"], json={**payload["json"], "stream": stream}
                    )
                    if stream:
                        assert response.status_code == 200
                        chunks = [
                            line.removeprefix("data: ")
                            for line in response.text.splitlines()
                            if line.startswith("data: ")
                        ]
                        assert chunks[-1] == "[DONE]"
                        responses.append(json.loads(chunks[0]))
                    else:
                        assert response.status_code == 400
                        responses.append(response.json())
        return completed_trial(1.0, "ContextLengthExceededError")

    session = HarborTrialSession(run, "reward")
    try:
        await session.prepare()
        turn = ModelTurn({"role": "assistant", "content": "edit"}, (2,), (3,), (-0.2,), "stop")
        assert not (await session.advance(turn)).done
        async with asyncio.timeout(5):
            grade = await session.grade((), stop_reason="length", outputs={})
        assert grade.status == Outcome.GRADED and grade.reward == 1.0
        assert len(responses) == 2
        assert all(response["error"]["code"] == 400 for response in responses)
        assert all(response["error"]["type"] == "context_length_exceeded" for response in responses)
    finally:
        await session.close()
        set_global_state(None, None)
