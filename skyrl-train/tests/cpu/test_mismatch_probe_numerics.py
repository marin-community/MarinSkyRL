"""Probe modes score from the trainer's own numerics when training runs under a numerics stack."""

from types import SimpleNamespace

import pytest

from skyrl_train.mismatch_probe.modes import NUMERICS_CANDIDATES, probe_mode_scope
from skyrl_train.mismatch_probe.numerics import GrugNumerics, active_numerics, set_default_numerics


@pytest.fixture
def training_under_compiled_stack():
    set_default_numerics(NUMERICS_CANDIDATES["compiled_stack"])
    yield GrugNumerics(**NUMERICS_CANDIDATES["compiled_stack"])
    set_default_numerics({})


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("native", GrugNumerics()), ("native+gated_norm", GrugNumerics(gated_norm=True))],
)
def test_probe_modes_ignore_the_training_numerics(training_under_compiled_stack, mode, expected):
    worker = SimpleNamespace(model=SimpleNamespace(router_replay=None))
    assert active_numerics() == training_under_compiled_stack
    with probe_mode_scope(worker, {"probe_mode": mode}):
        assert active_numerics() == expected
    assert active_numerics() == training_under_compiled_stack
