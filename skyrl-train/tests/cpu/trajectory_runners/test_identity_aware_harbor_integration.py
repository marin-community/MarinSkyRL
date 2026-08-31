from types import SimpleNamespace

import pytest
from skyrl_gym.verification import RewardResult

try:
    from skyrl_train.trajectory_runners.harbor.configuration import REWARD_SHAPING_SCHEMA
    from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)

from skyrl_train.metric_names import IDENTITY_AWARE_REWARD_METRIC_PREFIX
from skyrl_train.trajectory_runners.harbor.identity_aware_reward import IDENTITY_AWARE_SHAPER
from skyrl_train.trajectory_runners.types import TrajectoryID, VerifierTestCollection


def _collection(trial: int, outcomes: dict[str, str]) -> VerifierTestCollection:
    return {
        "parser": "test",
        "complete": True,
        "tests": [
            {
                "record_id": f"trial-{trial}:{test_id}",
                "trial_id": TrajectoryID(instance_id="task", repetition_id=trial),
                "test_id": test_id,
                "outcome": outcome,
                "output": f"{test_id}: {outcome}",
            }
            for test_id, outcome in outcomes.items()
        ],
    }


def _runner_output(
    trial: int,
    outcomes: dict[str, str],
    aggregate_reward: float,
    *,
    truncation_penalized: bool = False,
):
    return SimpleNamespace(
        trajectory_id=TrajectoryID(instance_id="task", repetition_id=trial),
        verifier_tests=_collection(trial, outcomes),
        disposition=SimpleNamespace(baseline_eligible=True),
        reward_result=RewardResult(unshaped_reward=aggregate_reward, optimization_reward=aggregate_reward),
        truncation_penalized=truncation_penalized,
    )


def _runner(shaper: str | None = None):
    runner = object.__new__(HarborTrajectoryRunner)
    runner._reward_shaping_enabled = True
    runner._reward_shaping_config = {"shaper_kwargs": {}}
    if shaper is not None:
        runner._reward_shaping_config["reward_shaper"] = shaper
    runner._test_weight_filter = None
    runner._truncation_penalty = 0.25
    return runner


def test_harbor_runner_applies_identity_aware_shaping_as_the_default():
    outputs = [
        _runner_output(0, {"uniform": "passed", "mixed": "passed"}, 1.0),
        _runner_output(1, {"uniform": "passed", "mixed": "failed"}, 0.5),
    ]

    metrics = _runner()._apply_identity_aware_reward_shaping(outputs)

    assert [output.reward_result.optimization_reward for output in outputs] == [1.0, 0.0]
    assert metrics[f"{IDENTITY_AWARE_REWARD_METRIC_PREFIX}/groups"] == 1
    assert REWARD_SHAPING_SCHEMA.fields["reward_shaper"].default == IDENTITY_AWARE_SHAPER


def test_legacy_pass_ratio_is_an_explicit_backup_mode():
    outputs = [
        _runner_output(0, {"uniform": "passed", "mixed": "passed"}, 1.0),
        _runner_output(1, {"uniform": "passed", "mixed": "failed"}, 0.5),
    ]

    metrics = _runner("pass_ratio")._apply_identity_aware_reward_shaping(outputs)

    assert [output.reward_result.optimization_reward for output in outputs] == [1.0, 0.5]
    assert metrics == {}


def test_identity_aware_shaping_preserves_the_downstream_truncation_penalty():
    outputs = [
        _runner_output(0, {"mixed": "passed"}, 0.75, truncation_penalized=True),
        _runner_output(1, {"mixed": "failed"}, 0.0),
    ]

    _runner()._apply_identity_aware_reward_shaping(outputs)

    assert [output.reward_result.optimization_reward for output in outputs] == [0.75, 0.0]
