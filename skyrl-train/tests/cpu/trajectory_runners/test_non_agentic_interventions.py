from dataclasses import replace
import random

import pytest

from skyrl_train.trajectory_runners.non_agentic_interventions import (
    INTERVENTION_VERSION,
    InterventionState,
    TokenIntervention,
    intervention_trace,
    repeated_ngram_fraction,
)


def config(kind="force_close", **kwargs):
    return TokenIntervention(INTERVENTION_VERSION, kind, 128003, 128009, **kwargs)


def test_force_close_at_exact_sampled_boundary_and_excludes_only_forced_action():
    c = config()
    tokens = list(range(3072))
    state = InterventionState(c)
    assert state.advance(tokens[:-1]) is None
    assert state.advance(tokens) == 128003
    tokens += [128003, 7, 8, 128009]
    assert state.advance(tokens) is None
    trace = intervention_trace(tokens, c)
    assert trace["forced_positions"] == [3072]
    assert trace["sampled_token_count"] == len(tokens) - 1
    assert trace["sampled_positions"] == list(range(3072)) + [3073, 3074, 3075]
    assert trace["thinking_closed"] and not trace["repetition_stopped"]


def test_natural_closure_never_forced_and_boundary_wrong_token_rejected():
    c = config(force_close_after=4)
    trace = intervention_trace([1, 128003, 2, 3, 4], c)
    assert trace["forced_positions"] == []
    assert trace["sampled_token_count"] == 5
    with pytest.raises(ValueError, match="prescribed forced token"):
        intervention_trace([1, 2, 3, 4, 5], c)


def test_repetition_first_check_at_256_then_forced_eos_is_not_sampled():
    c = config("repetition_stop")
    state = InterventionState(c)
    assert state.advance([4] * 255) is None
    assert state.advance([4] * 256) == 128009
    assert repeated_ngram_fraction([4] * 256, 16) == 240 / 241
    trace = intervention_trace([4] * 256 + [128009], c)
    assert trace["repetition_stopped"]
    assert trace["forced_positions"] == [256]
    assert trace["sampled_token_count"] == 256
    with pytest.raises(ValueError, match="continued"):
        intervention_trace([4] * 256 + [128009, 5], c)


def test_repetition_rolling_window_matches_stage1_reference_until_first_stop():
    rng = random.Random(17)
    tokens = [rng.randrange(1000) for _ in range(600)] + [8] * 400
    c = config("repetition_stop")
    state = InterventionState(c)
    for length in range(1, len(tokens) + 1):
        actual = state.advance(tokens[:length])
        expected = length >= 256 and repeated_ngram_fraction(tokens[max(0, length - 256) : length], 16) >= 0.5
        assert (actual == 128009) == expected
        if expected:
            assert length > 600
            break
    else:
        pytest.fail("repeated suffix did not trigger")


def test_native_prefix_mutation_rejected():
    state = InterventionState(config())
    state.advance([1, 2, 3])
    with pytest.raises(ValueError, match="changed"):
        state.advance([1, 5, 3, 4])


@pytest.mark.parametrize(
    "change",
    [
        {"protocol": "other"},
        {"kind": "other"},
        {"eos_id": 128003},
        {"repetition_fraction": 0},
        {"force_close_after": 0},
    ],
)
def test_invalid_protocol_rejected(change):
    with pytest.raises(ValueError):
        replace(config(), **change)
