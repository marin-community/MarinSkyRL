from types import SimpleNamespace

import pytest
import torch

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.distributed.megatron import megatron_utils  # noqa: E402


@pytest.mark.parametrize(
    ("cp_rank", "expected_tokens"),
    [
        (0, [17, 0]),
        (1, [0, 0]),
    ],
)
def test_preprocess_packed_seqs_short_sequence_preserves_rank_tokens(monkeypatch, cp_rank, expected_tokens):
    monkeypatch.setattr(megatron_utils.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(megatron_utils.mpu, "get_context_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(megatron_utils.mpu, "get_context_parallel_rank", lambda: cp_rank, raising=False)
    monkeypatch.setattr(megatron_utils, "PackedSeqParams", SimpleNamespace)

    input_ids = torch.tensor([[17, 23, 29, 31]])
    attention_mask = torch.tensor([[True, False, False, False]])

    packed_tokens, packed_seq_params = megatron_utils.preprocess_packed_seqs(input_ids, attention_mask)

    assert packed_tokens.tolist() == [expected_tokens]
    assert packed_seq_params.cu_seqlens_q.tolist() == [0, 4]
