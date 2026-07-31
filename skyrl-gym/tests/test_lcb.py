import multiprocessing
import pytest
import skyrl_gym
import json
from omegaconf import DictConfig

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
def test_compute_score(model_response, tests, expected_reward):
    env = skyrl_gym.make(
        "lcb",
        env_config=DictConfig({"env_class": "lcb"}),
        extras={"reward_model": {"method": "rule", "ground_truth": tests}},
    )
    # Skip init() since it's not used in this test
    step_output = env.step(model_response)
    assert step_output["reward"] == expected_reward


@pytest.fixture
def spawn_start_method():
    """Make `spawn` the default start method, as `skyrl_train.entrypoints.main_base` does."""
    original = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method("spawn", force=True)
    yield
    multiprocessing.set_start_method(original, force=True)


@pytest.mark.usefixtures("spawn_start_method")
def test_compute_score_under_spawn():
    """Scoring runs the tests in a child process, whose target `spawn` re-imports rather than inherits."""
    env = skyrl_gym.make(
        "lcb",
        env_config=DictConfig({"env_class": "lcb"}),
        extras={
            "reward_model": {
                "method": "rule",
                "ground_truth": json.dumps([{"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"}]),
            }
        },
    )
    assert env.step(SECOND_LARGEST_SOLUTION)["reward"] == 1.0


@pytest.mark.parametrize(
    "extras",
    [
        {},
        {"reward_model": {}},
        {"reward_model": {"ground_truth": "not JSON"}},
        {"reward_model": {"ground_truth": "[]"}},
    ],
)
def test_malformed_reward_model_scores_zero(extras, caplog):
    env = skyrl_gym.make(
        "lcb",
        env_config=DictConfig({"env_class": "lcb"}),
        extras=extras,
    )

    output = env.step(SECOND_LARGEST_SOLUTION)

    assert output["reward"] == 0.0
    assert output["metadata"]["verifier_error"] == "invalid reward_model.ground_truth"
    assert "invalid reward_model.ground_truth; scoring 0" in caplog.text
