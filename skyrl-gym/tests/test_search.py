"""Search task transitions through a real local HTTP retrieval service."""

import json

import pytest
from rolloutengine.contracts import RolloutInterrupted, RolloutOperation

from skyrl_gym.envs.search.utils import em_check, extract_solution


def config(url):
    return {"search_url": url, "topk": 3, "timeout": 5, "log_requests": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["France capital", "missing", "rejected", "malformed", "retry"])
async def test_search_failure_observations_allow_a_later_final_answer(rollout_session, retrieval_service, query):
    url, requests = retrieval_service
    rollout = await rollout_session(
        "search",
        [f"<search>{query}</search>", "<answer>Paris</answer>"],
        {"reward_spec": {"ground_truth": {"target": "Paris"}}},
        config(url),
    )
    observation = rollout.steps[0].transition.observations[0]
    content = json.loads(observation["content"].strip().removeprefix("<information>").removesuffix("</information>"))[
        "result"
    ]
    if query in {"rejected", "malformed"}:
        assert content.startswith("Search error:")
    elif query == "missing":
        assert content == "No search results found."
    else:
        assert content == "Doc 1: Paris is the capital of France.\n"
    assert requests == [{"query": query, "topk": 3, "return_scores": True}] * (2 if query == "retry" else 1)
    assert rollout.steps[1].messages[-2] == observation
    assert [step.transition.reward for step in rollout.steps] == [0.0, 1.0]
    assert rollout.grade.reward == 0.5
    assert rollout.loss_mask == (1, 1, 0, 0, 1, 1)


@pytest.mark.asyncio
async def test_invalid_search_action_does_not_enter_the_http_service(rollout_session, retrieval_service):
    url, requests = retrieval_service
    rollout = await rollout_session(
        "search",
        ["<search>no closing tag", "<answer>Paris</answer>"],
        {"reward_spec": {"ground_truth": {"target": "Paris"}}},
        config(url),
    )
    assert requests == []
    assert rollout.steps[0].transition.observations == ({"role": "user", "content": "\n<information></information>\n"},)
    assert rollout.steps[1].transition.reward == 1.0


@pytest.mark.asyncio
async def test_turn_limit_ends_search_without_another_http_request(rollout_session, retrieval_service):
    url, requests = retrieval_service
    rollout = await rollout_session(
        "search",
        ["<search>France capital</search>", "<search>More information</search>"],
        {"reward_spec": {"ground_truth": {"target": "Paris"}}},
        config(url),
        max_turns=2,
    )
    assert len(requests) == 1
    assert rollout.steps[-1].transition.done is True
    assert rollout.steps[-1].transition.observations == ()
    assert rollout.grade.reward == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,reward",
    [
        ("<answer>Paris</answer>", 1),
        ("<answer>  PARIS!  </answer>", 1),
        ("<answer>Rome</answer>", 0),
        ("No answer tags", 0),
    ],
)
async def test_search_task_preserves_exact_answer_normalization(rollout_session, retrieval_service, response, reward):
    url, requests = retrieval_service
    rollout = await rollout_session(
        "search",
        [response],
        {"reward_spec": {"ground_truth": {"target": ["The Paris", "Paris"]}}},
        config(url),
        max_turns=1,
    )
    assert rollout.grade.reward == reward
    assert requests == []


@pytest.mark.asyncio
async def test_search_rejects_a_response_beyond_its_stop_tag(rollout_session, retrieval_service):
    url, requests = retrieval_service
    with pytest.raises(RolloutInterrupted) as failure:
        await rollout_session(
            "search",
            ["<search>query</search>more text"],
            {"reward_spec": {"ground_truth": {"target": "Paris"}}},
            config(url),
        )
    assert failure.value.operation == RolloutOperation.ADVANCE
    assert requests == []


@pytest.mark.parametrize(
    "response,answer",
    [
        ("<answer>Paris</answer>", "Paris"),
        ("<answer>wrong</answer><answer> Paris </answer>", "Paris"),
        ("<answer>Paris", None),
        ("<answer></answer>", ""),
    ],
)
def test_search_answer_extraction_uses_the_last_completed_tag(response, answer):
    assert extract_solution(response) == answer


@pytest.mark.parametrize(
    "prediction,answers,reward", [("PARIS!", "The Paris", 1), ("Paris", ["London", "Paris"], 1), ("Rome", ["Paris"], 0)]
)
def test_search_exact_match_uses_case_punctuation_and_articles(prediction, answers, reward):
    assert em_check(prediction, answers) == reward
