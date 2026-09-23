import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.workers.megatron import megatron_model_wrapper as mmw  # noqa: E402
from tests.cpu.util import gloo_process_group  # noqa: E402

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


def _dense_logprob_worker(rank: int, port: int) -> None:
    with gloo_process_group(rank, 2, port):
        tp_groups = [dist.new_group([tp_rank]) for tp_rank in range(2)]
        mmw.mpu.get_tensor_model_parallel_group = lambda: tp_groups[rank]
        mmw.mpu.get_tensor_model_parallel_rank = lambda: 0
        mmw.mpu.get_tensor_model_parallel_world_size = lambda: 1
        mmw.mpu.get_context_parallel_group = lambda: dist.group.WORLD

        generator = torch.Generator().manual_seed(42)
        full_logits = torch.randn((1, 12, 16), generator=generator)
        sequences = torch.randint(0, 16, (1, 9), generator=generator)
        attention_mask = torch.ones_like(sequences, dtype=torch.bool)
        local_logits = shard_dense_sequence_for_context_parallel(full_logits, cp_size=2, cp_rank=rank)

        wrapper = mmw.MegatronModelWrapper.__new__(mmw.MegatronModelWrapper)
        wrapper.actor_module = [torch.nn.Identity().eval()]
        wrapper.use_sample_packing = False
        wrapper._logprob_chunk_size = 3

        actual = wrapper._token_logprobs(local_logits, sequences, attention_mask, None)
        expected = (
            torch.log_softmax(full_logits[:, :9, :], dim=-1)[:, :-1, :]
            .gather(-1, sequences[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
        torch.testing.assert_close(actual, expected)


def test_dense_context_parallel_gathers_logprobs_before_scattering(unused_tcp_port) -> None:
    mp.spawn(_dense_logprob_worker, args=(unused_tcp_port,), nprocs=2, join=True)
