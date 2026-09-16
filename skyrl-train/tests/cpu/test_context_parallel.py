import pytest
import torch

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.workers.megatron import megatron_model_wrapper as mmw  # noqa: E402
from skyrl_train.utils.context_parallel import shard_dense_sequence_for_context_parallel  # noqa: E402


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


def test_dense_context_parallel_gathers_logprobs_before_scattering(monkeypatch) -> None:
    """Reproduce the CP=2 forward shape that failed the Snowball canary."""
    cp_group = object()
    wrapper = mmw.MegatronModelWrapper.__new__(mmw.MegatronModelWrapper)
    wrapper.actor_module = [torch.nn.Identity()]
    wrapper.use_sample_packing = False
    wrapper._logprob_chunk_size = 1024

    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_group", lambda: object(), raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_group", lambda: cp_group, raising=False)

    def fake_logprobs(logits, target, **kwargs):
        assert logits.shape == (1, 2274, 8)
        assert target.shape == (1, 4545)
        assert kwargs["cp_group"] is cp_group
        # CP all-gather restores the full compact sequence and then drops the
        # final next-token target: 4,545 input tokens -> 4,544 logprobs.
        return torch.arange(4544, dtype=logits.dtype).unsqueeze(0)

    monkeypatch.setattr(mmw, "from_parallel_logits_to_logprobs", fake_logprobs)

    values = wrapper._token_logprobs(
        logits=torch.zeros((1, 2274, 8)),
        sequences=torch.arange(4545).unsqueeze(0),
        attention_mask=torch.ones((1, 4545), dtype=torch.bool),
        packed_seq_params=None,
    )

    assert values.shape == (1, 4544)
    torch.testing.assert_close(values, torch.arange(4544, dtype=values.dtype).unsqueeze(0))
