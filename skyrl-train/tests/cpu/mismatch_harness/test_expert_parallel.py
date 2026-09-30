import pytest
import torch

from skyrl_train.mismatch_harness.expert_parallel import ReduceOrder, reduce_partials

# bf16 keeps 8 significant bits, so 1 + 2**-8 is a tie that rounds back to 1 while 1 + 2**-7 is exact:
# whether the two half-ulp partials survive depends on where each addition rounds.
HALF_ULP = 2.0**-8


@pytest.mark.parametrize(
    ("order", "home_rank", "expected"),
    [
        (ReduceOrder.RANK, 0, 1.0),
        (ReduceOrder.RING, 0, 1.0 + 2 * HALF_ULP),
        (ReduceOrder.RING, 1, 1.0),
        (ReduceOrder.FP32, 1, 1.0 + 2 * HALF_ULP),
    ],
)
def test_reduce_models_round_where_they_add(order, home_rank, expected):
    partials = torch.tensor([1.0, HALF_ULP, HALF_ULP], dtype=torch.bfloat16).reshape(3, 1, 1)

    assert reduce_partials(partials, order, home_rank).item() == expected
