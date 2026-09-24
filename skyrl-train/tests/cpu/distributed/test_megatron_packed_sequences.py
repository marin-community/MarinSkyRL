from types import SimpleNamespace

import pytest
import torch

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.distributed.megatron import megatron_utils  # noqa: E402


def test_materialize_megatron_params_completes_deferred_gathers() -> None:
    deferred = megatron_utils.DDP()
    deferred.ddp_config = SimpleNamespace(overlap_param_gather=True)
    calls = []
    deferred.start_param_sync = lambda *, force_sync: calls.append(force_sync)
    synchronous = megatron_utils.DDP()
    synchronous.ddp_config = SimpleNamespace(overlap_param_gather=False)
    synchronous.start_param_sync = lambda *, force_sync: calls.append(force_sync)

    megatron_utils.materialize_megatron_params([deferred, synchronous, torch.nn.Linear(1, 1)])

    assert calls == [True]


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

    packed_tokens, _ = megatron_utils.preprocess_packed_seqs(input_ids, attention_mask)

    assert packed_tokens.tolist() == [expected_tokens]
