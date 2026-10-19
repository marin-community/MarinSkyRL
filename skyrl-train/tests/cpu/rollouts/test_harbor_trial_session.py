import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import httpx
import pytest
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import GenerationLimitReached, ModelTurn, Prompt
from taskcompendium.grading_result import Outcome
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from skyrl_train.rollouts.agent_backend import AgentInferenceBackend, TRIAL_HEADER
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.rollouts.harbor_settings import HARBOR_EXCEPTION
from skyrl_train.rollouts.harbor_tasks import harbor_grading_failure
from skyrl_train.rollouts.harbor_trial_session import HarborTrialHarness, run_harbor_trial, trial_grade
from skyrl_train.rollouts.training_record import TrainingRollout, training_rollout
from skyrl_train.rollouts.task_projections import rollout_loss_eligible
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig
from tests.cpu.task_specs import lowered_task, text_task


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
        TrainingRollout(
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
async def test_trial_harness_preserves_turns_and_returns_original_verifier_score():
    delivered = []

    async def run(complete):
        history = ({"role": "user", "content": "Repair the repository."},)
        first = await complete(Prompt(history, {}))
        delivered.append(first)
        second = await complete(Prompt((*history, first.message, {"role": "user", "content": "Tests passed."}), {}))
        delivered.append(second)
        return completed_trial(1.0)

    first = ModelTurn({"role": "assistant", "content": "edit"}, (11,), (12,), (-0.3,), "stop")
    second = ModelTurn({"role": "assistant", "content": "done"}, (11, 12, 13), (14,), (-0.2,), "stop")
    responses = iter((first, second))

    async def model(request):
        return next(responses)

    spec = lowered_task(text_task(), session_name="harbor_trial", max_turns=3)
    engine = RolloutEngine({}, {"harbor_trial": HarborTrialHarness(run, "reward")})
    result = training_rollout(await engine.run_task(spec, model), spec)
    assert result.grade.status == Outcome.GRADED and result.grade.reward == 1.0
    assert [step.turn for step in result.steps] == [first, second]
    assert delivered == [first, second]


@pytest.mark.asyncio
async def test_trial_harness_cancellation_finishes_trial_cleanup():
    cleaned = asyncio.Event()
    started = asyncio.Event()

    async def run(complete):
        started.set()
        try:
            await complete(Prompt(({"role": "user", "content": "Task"},), {}))
        finally:
            cleaned.set()

    async def model(request):
        await asyncio.Future()

    spec = lowered_task(text_task(), session_name="harbor_trial")
    engine = RolloutEngine({}, {"harbor_trial": HarborTrialHarness(run, "reward")})
    attempt = asyncio.create_task(engine.run_task(spec, model))
    await started.wait()
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    assert cleaned.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_trial_harness_context_limit_preserves_verification(stream):
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
            payload["json"]["messages"] += [
                {"role": "assistant", "content": "edit"},
                {"role": "user", "content": "Continue"},
            ]
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
                        responses.append(json.loads(chunks[0]))
                    else:
                        assert response.status_code == 400
                        responses.append(response.json())
        return completed_trial(1.0, "ContextLengthExceededError")

    calls = 0

    async def model(request):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise GenerationLimitReached((2, 3))
        return ModelTurn({"role": "assistant", "content": "edit"}, (2,), (3,), (-0.2,), "stop")

    spec = lowered_task(text_task(), session_name="harbor_trial")
    engine = RolloutEngine({}, {"harbor_trial": HarborTrialHarness(run, "reward")})
    try:
        result = training_rollout(await asyncio.wait_for(engine.run_task(spec, model), 5), spec)
        assert result.grade.status == Outcome.GRADED and result.grade.reward == 1.0
        assert len(responses) == 2
        assert all(response["error"]["type"] == "context_length_exceeded" for response in responses)
    finally:
        set_global_state(None, None)
