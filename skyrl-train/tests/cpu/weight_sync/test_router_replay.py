import weakref

import pytest
import torch

from skyrl_train.weight_sync.byte_replay import compare_installed_views
from skyrl_train.weight_sync.worker_bucket_protocol import REPLAY_SCRATCH_BYTES

from skyrl_train.weight_sync.router_replay import ROUTER_CONVERSION_ELEMENTS, compare_widened_router
from skyrl_train.weight_sync.manifest import TensorSpec, build_manifest, pack_bucket
from skyrl_train.weight_sync.bucket_receiver import GrugBucketReceiver


@pytest.mark.parametrize("byte_offset", [0, 1, 2, 3, 4095])
def test_router_replay_detects_every_fp32_byte_including_zero_low_half(byte_offset):
    source = torch.arange(1024).bfloat16()
    installed = source.float()
    installed.view(torch.uint8)[byte_offset] ^= 1
    result = compare_widened_router(source, installed, torch.empty(97, dtype=torch.bool))
    assert result.compared_bytes == 4096 and result.mismatches == 1


def test_conversion_lifetime_is_bounded_between_chunks(monkeypatch):
    source = torch.arange(2 * ROUTER_CONVERSION_ELEMENTS + 1).bfloat16()
    installed = source.float()
    original = torch.Tensor.to
    previous = None
    sizes = []

    def observed(tensor, *args, **kwargs):
        nonlocal previous
        if tensor.dtype == torch.bfloat16 and args == (torch.float32,):
            assert previous is None or previous() is None, "old conversion is still allocated"
            result = original(tensor, *args, **kwargs)
            sizes.append(result.numel() * result.element_size())
            previous = weakref.ref(result)
            return result
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observed)
    result = compare_widened_router(source, installed, torch.empty(512 * 1024, dtype=torch.bool))
    assert result.mismatches == 0 and result.compared_bytes == installed.numel() * 4
    assert sizes == [256 * 1024, 256 * 1024, 4]
    assert previous() is None


def test_aliases_reject_before_any_comparison():
    backing = torch.zeros(8, dtype=torch.uint8)
    source = backing.view(torch.bfloat16)[:2]
    installed = backing.view(torch.float32)
    with pytest.raises(ValueError, match="must not overlap"):
        compare_widened_router(source, installed, torch.empty(8, dtype=torch.bool))
    source = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must not overlap"):
        compare_widened_router(source, installed, source.view(torch.bool))


def test_actual_bucket_router_install_and_replay_count_installed_fp32_bytes():
    name = "model.layers.0.mlp.router.weight"
    manifest = build_manifest([TensorSpec(name, (2, 4), "bfloat16")], bucket_bytes=32)
    source = torch.arange(8).bfloat16().reshape(2, 4)
    installed = torch.full((2, 4), -1.0, dtype=torch.float32)
    buffers = (torch.empty(32, dtype=torch.uint8), torch.empty(32, dtype=torch.uint8))
    receiver = GrugBucketReceiver(manifest, {name: installed}, {}, buffers, backend="TRITON", tensor_parallel_size=1)
    assert receiver.expected_bytes == 32
    pack_bucket(manifest, 0, {name: source}, buffers[0])
    assert receiver.install_bucket(0) == 32
    assert torch.equal(installed.view(torch.int32), source.float().view(torch.int32))
    result = receiver.replay_bucket(0, torch.empty(16, dtype=torch.bool))
    assert result.compared_bytes == 32 and result.mismatches == 0
    assert receiver.finish_replay() == result


def test_other_dense_dtype_mismatch_remains_rejected():
    name = "model.embed_tokens.weight"
    manifest = build_manifest([TensorSpec(name, (2, 4), "bfloat16")], bucket_bytes=32)
    parameters = {name: torch.empty(2, 4, dtype=torch.float32)}
    buffers = (torch.empty(32, dtype=torch.uint8), torch.empty(32, dtype=torch.uint8))
    with pytest.raises(ValueError, match="name=model.embed_tokens.weight"):
        GrugBucketReceiver(manifest, parameters, {}, buffers, backend="TRITON", tensor_parallel_size=1)


def test_production_scratch_preserves_exact_mismatch_count_across_chunks():
    scratch = torch.empty(REPLAY_SCRATCH_BYTES, dtype=torch.bool)
    source = torch.zeros(REPLAY_SCRATCH_BYTES * 3 + 1, dtype=torch.uint8)
    installed = source.clone()
    installed[REPLAY_SCRATCH_BYTES - 1] = 1
    installed[REPLAY_SCRATCH_BYTES] = 1
    installed[-1] = 1
    result = compare_installed_views(((source, installed),), scratch, expected_bytes=source.numel())
    assert result.compared_bytes == source.numel() and result.mismatches == 3
