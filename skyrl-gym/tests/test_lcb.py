import multiprocessing

import pytest
import json
from taskcompendium.grading import Outcome

SECOND_LARGEST_SOLUTION = """```python
def main():
    N = int(input())
    A = list(map(int, input().split()))
    B = sorted(A, reverse=True)
    second = B[1]
    print(A.index(second) + 1)

if __name__ == "__main__":
    main()
```"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_response, tests, expected_reward",
    [
        # Correct code: second largest index
        (
            SECOND_LARGEST_SOLUTION,
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                    {"input": "3\n3 2 1\n", "output": "2\n", "testtype": "stdin"},
                ]
            ),
            1.0,
        ),
        # Correct code reading stdin through the bytes view
        (
            """```python
import sys
print(sum(map(int, sys.stdin.buffer.read().split())))
```""",
            json.dumps(
                [
                    {"input": "4 7\n", "output": "11\n", "testtype": "stdin"},
                    {"input": "2 3\n", "output": "5\n", "testtype": "stdin"},
                ]
            ),
            1.0,
        ),
        # Wrong logic: returns index of largest
        (
            """```python
def main():
    N = int(input())
    A = list(map(int, input().split()))
    print(A.index(max(A)) + 1)

if __name__ == "__main__":
    main()
```""",
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                ]
            ),
            0.0,
        ),
        # Missing main() call — runtime error
        (
            """```python
def main():
    A = list(map(int, input().split()))
    B = sorted(A, reverse=True)
    second = B[1]
    print(A.index(second) + 1)

# forgot to call main()
```""",
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                ]
            ),
            0.0,
        ),
    ],
)
async def test_task_executes_candidate_code_in_shellbox(rollout_session, model_response, tests, expected_reward):
    rollout = await rollout_session("lcb", [model_response], {"reward_model": {"ground_truth": tests}})
    assert rollout.grade.reward == expected_reward


@pytest.fixture
def spawn_start_method():
    """Make `spawn` the default start method, as `skyrl_train.entrypoints.main_base` does."""
    original = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method("spawn", force=True)
    yield
    multiprocessing.set_start_method(original, force=True)


@pytest.mark.usefixtures("spawn_start_method")
@pytest.mark.asyncio
async def test_task_execution_remains_valid_when_the_trainer_uses_spawn(rollout_session):
    rollout = await rollout_session(
        "lcb",
        [SECOND_LARGEST_SOLUTION],
        {
            "reward_model": {
                "method": "rule",
                "ground_truth": json.dumps([{"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"}]),
            }
        },
    )
    assert rollout.grade.reward == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extras",
    [
        {},
        {"reward_model": {}},
        {"reward_model": {"ground_truth": "not JSON"}},
        {"reward_model": {"ground_truth": "[]"}},
        {"reward_model": {"ground_truth": "[{}]"}},
        {"reward_model": {"ground_truth": "[1]"}},
    ],
)
async def test_malformed_reward_model_has_no_grade(rollout_session, extras):
    rollout = await rollout_session("lcb", [SECOND_LARGEST_SOLUTION], extras)
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.INFRA_ERROR, None)
    assert "verifier_error" in rollout.steps[0].transition.metrics


@pytest.mark.asyncio
async def test_fractional_reward_reports_fraction_of_tests_passed_while_binary_remains_all_or_nothing(rollout_session):
    tests = json.dumps(
        [
            {"input": "1\n", "output": "1\n", "testtype": "stdin"},
            {"input": "2\n", "output": "2\n", "testtype": "stdin"},
            {"input": "3\n", "output": "6\n", "testtype": "stdin"},
            {"input": "4\n", "output": "8\n", "testtype": "stdin"},
        ]
    )
    response = """```python
print(int(input()))
```"""

    for mode, reward in [("fractional", 0.5), ("binary", 0.0)]:
        rollout = await rollout_session(
            "lcb", [response], {"reward_model": {"ground_truth": tests}}, {"reward_mode": mode}
        )
        assert rollout.grade.reward == reward
