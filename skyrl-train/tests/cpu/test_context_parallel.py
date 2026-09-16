import pytest
import torch
from skyrl_train.utils.context_parallel import shard_dense_sequence_for_context_parallel


@pytest.mark.parametrize(
    ("cp_rank", "expected"),
    [
        (0, [10, 11, 16, 0]),
        (1, [12, 13, 14, 15]),
    ],
)
def test_dense_context_parallel_uses_load_balanced_causal_chunks(cp_rank: int, expected: list[int]) -> None:
    padded_sequence = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 0]])

    local_sequence = shard_dense_sequence_for_context_parallel(padded_sequence, cp_size=2, cp_rank=cp_rank)

    assert local_sequence.tolist() == [expected]


def test_dense_context_parallel_rejects_unaligned_sequence() -> None:
    with pytest.raises(ValueError, match=r"must be divisible by 2 \* cp_size"):
        shard_dense_sequence_for_context_parallel(torch.arange(7).unsqueeze(0), cp_size=2, cp_rank=0)
