"""The parity comparator must expose a changed optimizer or RNG leaf."""

import numpy as np
import pytest
import torch

from tests.checkpoint_parity_state import assert_state_equal, count_tensors, snapshot_value


def test_snapshot_is_independent_and_comparator_detects_numerical_drift():
    live = {"optimizer": {"exp_avg": torch.tensor([0.1, 0.2])}, "rng": np.array([1, 2], dtype=np.uint32)}
    expected = snapshot_value(live)
    assert count_tensors(expected) == 2
    assert_state_equal(expected, snapshot_value(live))

    live["optimizer"]["exp_avg"][1] += 0.01
    with pytest.raises(AssertionError, match=r"optimizer.*exp_avg.*max_abs"):
        assert_state_equal(expected, snapshot_value(live))

    live["optimizer"]["exp_avg"][1] -= 0.01
    live["rng"][0] += 1
    with pytest.raises(AssertionError, match=r"rng.*max_abs"):
        assert_state_equal(expected, snapshot_value(live))
