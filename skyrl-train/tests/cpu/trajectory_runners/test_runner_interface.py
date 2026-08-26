"""`RolloutDispatcher` stands in for a `TrajectoryRunner` without inheriting from one.

`trainer.py` and `evaluate.py` annotate the concrete ABC and call whatever the fan-out swap left
behind, so nothing checks that the substitute still accepts the same calls. That gap has already
shipped twice: `set_trajectory_sink` was missing outright, which silently discarded every retained
trajectory, and `start_eval_session` was positional where the ABC declares it keyword-only.
"""

import inspect

import pytest

from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.harbor.rollout_dispatcher import RolloutDispatcher

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
