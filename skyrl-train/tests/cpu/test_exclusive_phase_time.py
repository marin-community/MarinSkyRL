import pytest

from skyrl_train import telemetry
from skyrl_train.telemetry import exclusive_durations


def test_children_are_subtracted_from_their_parent():
    timings = {"step": 10.0, "generate": 6.0, "run_training": 3.0}
    assert exclusive_durations(timings) == {"step": 1.0, "generate": 6.0, "run_training": 3.0}


def test_self_times_sum_to_the_root_span():
    timings = {"step": 12.0, "generate": 5.0, "run_training": 4.0, "policy_train": 3.0}
    exclusive = exclusive_durations(timings)
    # policy_train nests under train_critic_and_policy, which this run did not record, so it
    # resolves up to run_training rather than being counted twice against the step.
    assert exclusive["run_training"] == pytest.approx(1.0)
    assert sum(exclusive.values()) == pytest.approx(timings["step"])


def test_a_span_resolves_to_the_nearest_recorded_container():
    # critic_train -> train_critic_and_policy -> run_training -> step. With only step recorded
    # above it, its cost must land against step, not vanish.
    exclusive = exclusive_durations({"step": 8.0, "critic_train": 5.0})
    assert exclusive == {"step": 3.0, "critic_train": 5.0}


def test_overlapping_children_floor_at_zero_rather_than_going_negative():
    # The async trainer overlaps siblings, so children can exceed the parent. A negative bar
    # would render as a gap in the stack and read as missing time.
    exclusive = exclusive_durations({"step": 4.0, "generate": 3.0, "run_training": 3.0})
    assert exclusive["step"] == 0.0
    assert min(exclusive.values()) >= 0.0


def _recorded(monkeypatch):
    records = []
    monkeypatch.setattr(
        telemetry.phase_duration, "record", lambda value, attributes: records.append({**attributes, "value": value})
    )
    return records


def test_unknown_spans_are_ignored_rather_than_stacked(monkeypatch):
    records = _recorded(monkeypatch)

    telemetry.record_step_timings({"step": 2.0, "something_new": 1.0}, step=7)

    assert [r["phase"] for r in records] == ["step"]


def test_work_beside_a_step_stacks_under_its_own_root(monkeypatch):
    """Checkpointing and eval are recorded alongside a step and contained by none of it, so
    stacking every span together reports the step as the sum of itself and the work beside it."""
    records = _recorded(monkeypatch)

    telemetry.record_step_timings(
        {"step": 10.0, "generate": 6.0, "save_checkpoints": 5.0, "cleanup_old_checkpoints": 1.0, "eval": 4.0},
        step=7,
    )

    stacked = {}
    for record in records:
        stacked[record["root"]] = stacked.get(record["root"], 0.0) + record["value"]
    assert stacked == {"step": 10.0, "save_checkpoints": 5.0, "eval": 4.0}
