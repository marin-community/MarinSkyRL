import ast
import inspect
import textwrap

import pytest

from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.timing_observability import phase_timing_observations, publish_step_timings


def test_overlapping_async_spans_remain_inclusive_instead_of_claiming_exclusive_time():
    observations = phase_timing_observations({"step": 4.0, "generate": 3.0, "run_training": 3.0})

    assert {item.name: item.duration_seconds for item in observations} == {
        "step": 4.0,
        "generate": 3.0,
        "run_training": 3.0,
    }
    assert {item.name: item.parent for item in observations} == {
        "step": None,
        "generate": "step",
        "run_training": "step",
    }


def test_unknown_spans_are_not_published():
    calls = []

    class _Sink:
        def publish(self, observations, step):
            calls.append((observations, step))

    publish_step_timings({"step": 2.0, "something_new": 1.0}, step=7, sinks=(_Sink(),))

    observations, step = calls[0]
    assert [item.name for item in observations] == ["step"]
    assert step == 7


def test_post_step_work_is_published_under_the_step_root():
    observations = phase_timing_observations(
        {"step": 10.0, "eval": 3.0, "save_checkpoints": 2.0, "cleanup_old_checkpoints": 0.5}
    )

    by_name = {item.name: item for item in observations}
    assert by_name["eval"].parent == "step"
    assert by_name["save_checkpoints"].parent == "step"
    assert by_name["cleanup_old_checkpoints"].parent == "save_checkpoints"
    assert {item.root for item in observations} == {"step"}


@pytest.mark.parametrize("trainer_class", [RayPPOTrainer, FullyAsyncRayPPOTrainer])
def test_step_end_callbacks_run_inside_production_step_timer(trainer_class):
    tree = ast.parse(textwrap.dedent(inspect.getsource(trainer_class._train_loop)))
    step_timer_blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "Timer"
            and item.context_expr.args
            and isinstance(item.context_expr.args[0], ast.Constant)
            and item.context_expr.args[0].value == "step"
            for item in node.items
        )
    ]

    assert step_timer_blocks
    assert all(
        any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_run_step_end_callbacks"
            for node in ast.walk(block)
        )
        for block in step_timer_blocks
    )
