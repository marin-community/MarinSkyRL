"""Serialized tasks through direct sessions, exact-token execution, and private grading."""

import json

import pytest
from rolloutengine.contracts import ModelTurn
from rolloutengine.engine import ShellboxRolloutEngine
from taskcompendium.grading_result import Outcome
from rolloutengine.spec import LoweredTaskSpec, TaskRuntimeSpec, TaskSessionSpec
from skyrl_gym.source_task import source_task
from taskcompendium.models import Source
from taskcompendium.submission import PlainText

from skyrl_gym.task_factories import session_factories
from skyrl_gym.task_sessions import CodeTaskSession


class ReplayModel:
    def __init__(self, responses, *, stop_reason="stop", tokens=2, metadata=None):
        self.responses = iter(responses)
        self.requests = []
        self.stop_reason = stop_reason
        self.tokens = tokens
        self.metadata = {} if metadata is None else metadata

    async def __call__(self, request):
        self.requests.append(request)
        text = next(self.responses)
        prompt = (*request.prefix_token_ids, 90, 91) if request.prefix_token_ids else (10, 11)
        return ModelTurn(
            {"role": "assistant", "content": text},
            prompt,
            tuple(range(20, 20 + self.tokens)),
            (-0.5,) * self.tokens,
            self.stop_reason,
            text=text,
            metadata=self.metadata,
        )


def engine(model):
    return ShellboxRolloutEngine(
        model,
        {},
        convention=PlainText(id="plain"),
        sessions=session_factories(),
    )


def task(name, extras, config=None):
    specification = source_task(
        [{"role": "user", "content": "Public task question"}],
        extras,
        {} if config is None else config,
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    return LoweredTaskSpec(
        task=specification,
        runtime=TaskRuntimeSpec(task_machine=None, verifier_machine=None),
        session=TaskSessionSpec(
            task_session=name,
            max_turns=3,
            model_turn_timeout=None,
            command_timeout=1,
            tool_turn_timeout=5,
            total_turn_timeout=None,
            attempt_timeout=None,
            verifier_timeout=5,
            cleanup_timeout=5,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,extras,response,native_reward,training_reward",
    [
        ("aime", {"reward_model": {"ground_truth": "42"}}, "Answer: \\boxed{43}", -1.0, -1.0),
        ("gsm8k", {"reward_model": {"ground_truth": "12"}}, "#### 12", 1.0, 1.0),
        ("cat_count", {"extra_info": {"n": 4}}, "cat cat cat", 0.0, 0.4725),
        ("mcq", {"reward_model": {"ground_truth": "B"}}, "\\boxed{B}", 1.0, 1.0),
    ],
)
async def test_answer_task_json_preserves_private_grade_and_training_reward(
    name, extras, response, native_reward, training_reward
):
    specification = LoweredTaskSpec.model_validate_json(task(name, extras).model_dump_json())
    model = ReplayModel([response])
    rollout = await engine(model).run(specification)
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, native_reward)
    assert rollout.steps[0].transition.reward == pytest.approx(training_reward)
    assert rollout.prompt_token_ids == (10, 11)
    assert rollout.response_token_ids == (20, 21)
    assert rollout.loss_mask == (1, 1)
    assert rollout.logprobs == (-0.5, -0.5)
    assert model.requests[0].messages == ({"role": "user", "content": "Public task question"},)


@pytest.mark.asyncio
async def test_aime_session_preserves_native_grade_under_length_shaping():
    model = ReplayModel(["Answer: \\boxed{42}"], tokens=6, metadata={"generation_token_budget": 6})
    rollout = await engine(model).run(
        task(
            "aime",
            {"reward_model": {"ground_truth": "42"}},
            {"length_penalty_weight": 0.5, "target_length": 2, "min_response_length": 0, "evaluation_token_budget": 4},
        ),
    )
    assert rollout.grade.reward == 1.0
    assert rollout.grade.passed is True
    assert rollout.steps[0].transition.reward == pytest.approx(0.5)
    assert rollout.steps[0].transition.reward_components == {"length": pytest.approx(-0.5)}
    assert rollout.metrics["over_evaluation_budget"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason,reward", [("stop", 1.0), ("length", 0.0)])
async def test_final_line_requires_a_completed_model_response(stop_reason, reward):
    model = ReplayModel(["#### 12"], stop_reason=stop_reason)
    rollout = await engine(model).run(
        task("gsm8k", {"reward_model": {"ground_truth": "12"}}, {"reward_method": "final_line"}),
    )
    assert rollout.grade.reward == reward


@pytest.mark.asyncio
async def test_math_correction_preserves_per_turn_credit_and_masks_only_observation_tokens():
    model = ReplayModel(["#### 11", "#### 12"])
    rollout = await engine(model).run(task("gsm8k_multi_turn", {"reward_spec": {"ground_truth": "12"}}))
    assert [step.transition.reward for step in rollout.steps] == pytest.approx([0.2 / 3, 1.0])
    assert rollout.grade.reward == 1.0
    assert rollout.grade.passed is True
    assert [step.transition.grade.reward for step in rollout.steps] == [0.0, 1.0]
    assert rollout.response_token_ids == (20, 21, 90, 91, 20, 21)
    assert rollout.loss_mask == (1, 1, 0, 0, 1, 1)
    assert model.requests[1].messages[-1]["role"] == "user"
    assert rollout.metrics == {"steps": 2}


@pytest.mark.asyncio
async def test_math_format_rewards_do_not_produce_a_correctness_score():
    model = ReplayModel(["#### 11", "#### 10", "#### 9"])
    rollout = await engine(model).run(task("gsm8k_multi_turn", {"reward_spec": {"ground_truth": "12"}}))
    assert [step.transition.reward for step in rollout.steps] == pytest.approx([0.2 / 3] * 3)
    assert [step.transition.grade.reward for step in rollout.steps] == [0.0] * 3
    assert rollout.grade.reward == 0.0
    assert rollout.grade.passed is False


@pytest.mark.asyncio
async def test_search_observation_reaches_model_without_entering_the_loss_mask(retrieval_service):
    url, requests = retrieval_service
    model = ReplayModel(["<search>France capital</search>", "<answer>Paris</answer>"])
    rollout = await engine(model).run(
        task(
            "search",
            {"reward_spec": {"ground_truth": {"target": ["Paris"]}}},
            {"search_url": url, "topk": 3, "timeout": 5, "log_requests": False},
        ),
    )
    assert requests == [{"query": "France capital", "topk": 3, "return_scores": True}]
    observation = model.requests[1].messages[-1]
    assert observation["role"] == "user"
    assert json.loads(observation["content"].strip().removeprefix("<information>").removesuffix("</information>")) == {
        "result": "Doc 1: Paris is the capital of France.\n"
    }
    assert rollout.grade.reward == 1.0
    assert rollout.grade.passed is True
    assert rollout.steps[0].transition.grade.status is Outcome.UNAVAILABLE
    assert [step.transition.reward for step in rollout.steps] == [0.0, 1.0]
    assert rollout.loss_mask == (1, 1, 0, 0, 1, 1)


@pytest.mark.asyncio
async def test_lcb_machine_failure_reports_infrastructure_without_a_grade(machine, model_turn, monkeypatch):
    session = CodeTaskSession(
        task("lcb", {"reward_model": {"ground_truth": [{"input": "", "output": "42", "testtype": "stdin"}]}}),
        machine,
    )

    async def failed_command(command):
        raise OSError("Machine transport disconnected")

    monkeypatch.setattr(machine, "run", failed_command)
    transition = await session.advance(model_turn("```python\nprint(42)\n```"))
    assert transition.done is True
    assert (transition.grade.status, transition.grade.reward) == (Outcome.INFRA_ERROR, None)
    assert transition.grade.diagnostics == {
        "error_type": "OSError",
        "error_message": "Machine transport disconnected",
    }
    assert await session.grade(()) == transition.grade


@pytest.mark.asyncio
async def test_lcb_programmer_failure_propagates_instead_of_producing_a_training_result(
    machine, model_turn, monkeypatch
):
    session = CodeTaskSession(
        task("lcb", {"reward_model": {"ground_truth": [{"input": "", "output": "42", "testtype": "stdin"}]}}),
        machine,
    )
    error = TypeError("Invalid machine call")

    async def invalid_command(command):
        raise error

    monkeypatch.setattr(machine, "run", invalid_command)
    with pytest.raises(TypeError) as failure:
        await session.advance(model_turn("```python\nprint(42)\n```"))
    assert failure.value is error
