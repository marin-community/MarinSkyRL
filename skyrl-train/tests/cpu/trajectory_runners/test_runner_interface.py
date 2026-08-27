import inspect

import pytest

from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.harbor.rollout_dispatcher import RETAINED_RUNNER_NAME, RolloutDispatcher

# What `evaluate.py` and the trainer call on whatever the runner slot holds.
SHARED_METHODS = ("run", "set_trajectory_sink", "start_eval_session", "stop_eval_session", "shutdown")


def call_shape(method):
    """The part of a signature a caller has to get right: names, passing style, and optionality."""
    return [
        (name, parameter.kind, parameter.default is not inspect.Parameter.empty)
        for name, parameter in inspect.signature(method).parameters.items()
    ]


@pytest.mark.parametrize("name", SHARED_METHODS)
def test_the_dispatcher_accepts_the_same_calls_as_the_runner_it_replaces(name):
    expected = getattr(TrajectoryRunner, name, None)
    assert expected is not None, f"{name} is no longer on TrajectoryRunner; update SHARED_METHODS"

    substitute = getattr(RolloutDispatcher, name, None)
    assert substitute is not None, f"RolloutDispatcher is missing {name}, which callers invoke on it"
    assert call_shape(substitute) == call_shape(expected)


def test_the_retained_name_still_matches_the_class_it_names():
    # harbor installs on linux only, so this runs in CI and skips locally.
    pytest.importorskip("harbor")
    from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner

    # The fan-out path stamps this literal while the single-process path stamps the class name,
    # and TrajectorySink.bind_runner raises when the two disagree.
    assert RETAINED_RUNNER_NAME == HarborTrajectoryRunner.__name__
