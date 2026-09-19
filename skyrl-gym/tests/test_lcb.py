import multiprocessing
import platform
import time

import pytest
import skyrl_gym
import json
from omegaconf import DictConfig

from skyrl_gym.envs.lcb.livecodebench import VerifierLimits, lcb_test_results

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
        {"reward_model": {"ground_truth": "[{}]"}},
        {"reward_model": {"ground_truth": "[1]"}},
    ],
)
def test_malformed_reward_model_scores_zero(extras):
    env = skyrl_gym.make(
        "lcb",
        env_config=DictConfig({"env_class": "lcb"}),
        extras=extras,
    )

    output = env.step(SECOND_LARGEST_SOLUTION)

    assert output["reward"] == 0.0
    assert output["metadata"]["verifier_error"]


def test_fractional_reward_reports_fraction_of_tests_passed_while_binary_remains_all_or_nothing():
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

    fractional = skyrl_gym.make(
        "lcb",
        env_config=DictConfig({"reward_mode": "fractional"}),
        extras={"reward_model": {"ground_truth": tests}},
    )
    binary = skyrl_gym.make(
        "lcb",
        env_config=DictConfig({"reward_mode": "binary"}),
        extras={"reward_model": {"ground_truth": tests}},
    )

    assert fractional.step(response)["reward"] == 0.5
    assert binary.step(response)["reward"] == 0.0


def test_verifier_child_hits_wall_clock_deadline_and_is_reaped():
    """A wall-clock cap bounds the join window; without it the formula allows (1+1)*50+5 = 105s."""
    tests = [
        {"input": "1\n", "output": "1\n", "testtype": "stdin"},
    ] * 50
    sleeper = "import time\nprint(int(input()))\ntime.sleep(5)"

    started = time.monotonic()
    results = lcb_test_results(
        tests,
        sleeper,
        timeout=1,
        limits=VerifierLimits(total_timeout_seconds=3),
    )
    elapsed = time.monotonic() - started

    assert results == [-1] * 50
    assert elapsed < 15
    assert multiprocessing.active_children() == []


@pytest.mark.skipif(platform.system() == "Darwin", reason="RLIMIT_AS is not enforced on macOS")
def test_verifier_child_allocating_past_the_memory_cap_scores_zero():
    tests = [{"input": "1\n", "output": "1\n", "testtype": "stdin"}]
    bomb = "print(int(input()))\nbytearray(2 * 1024**3)"

    results = lcb_test_results(
        tests,
        bomb,
        timeout=5,
        limits=VerifierLimits(max_memory_bytes=512 * 1024**2, total_timeout_seconds=60),
    )

    assert all(result is not True for result in results)
    assert multiprocessing.active_children() == []
