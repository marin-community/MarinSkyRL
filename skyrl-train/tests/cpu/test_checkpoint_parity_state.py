"""The parity comparator must expose a changed optimizer or RNG leaf."""

import numpy as np
import pytest
import torch

from skyrl_train.checkpoint_parity_digest import digest_value
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


def test_streamed_digest_detects_value_dtype_and_structure_changes():
    state = {"model": torch.arange(128, dtype=torch.bfloat16), "optimizer": [torch.tensor([0.1, 0.2])]}
    original = digest_value(state)
    assert original["tensor_count"] == 2
    assert original["tensor_bytes"] == 264
    assert digest_value({"optimizer": state["optimizer"], "model": state["model"]}) == original

    state["optimizer"][0][1] += 0.01
    assert digest_value(state)["sha256"] != original["sha256"]
    state["optimizer"][0][1] -= 0.01
    state["model"] = state["model"].to(torch.float32)
    assert digest_value(state)["sha256"] != original["sha256"]
    state["model"] = state["model"].to(torch.bfloat16)
    state["optimizer"].append(torch.tensor([0.0]))
    assert digest_value(state)["sha256"] != original["sha256"]


def test_streamed_digest_includes_scalar_and_empty_tensor_dtype():
    state = {"scalar": torch.tensor(1.0), "empty": torch.empty(0, dtype=torch.bfloat16)}
    original = digest_value(state)
    assert original["tensor_count"] == 2
    assert original["tensor_bytes"] == 4
    state["scalar"] = torch.tensor(2.0)
    assert digest_value(state)["sha256"] != original["sha256"]
    state["scalar"] = torch.tensor(1.0)
    state["empty"] = torch.empty(0, dtype=torch.float32)
    assert digest_value(state)["sha256"] != original["sha256"]
