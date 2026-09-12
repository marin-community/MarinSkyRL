import pytest
import torch

from skyrl_train.weight_sync.byte_replay import MAX_SCRATCH_BYTES, ReceiverByteCoverage, compare_installed_views
from skyrl_train.weight_sync.expert_scatter import grug_expert_views, scatter_grug_experts
from skyrl_train.weight_sync.manifest import TensorSpec, build_manifest, pack_bucket, unpack_bucket


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_every_byte_including_odd_tail_and_float_payloads(dtype):
    source = torch.arange(15, dtype=dtype)
    source.view(torch.uint8)[-1] = 255
    target = source.clone()
    scratch = torch.empty(3, dtype=torch.bool)
    size = source.numel() * source.element_size()
    assert compare_installed_views([(source, target)], scratch, expected_bytes=size).mismatches == 0
    for byte in range(size):
        target.copy_(source)
        target.view(torch.uint8)[byte] ^= 1
        result = compare_installed_views([(source, target)], scratch, expected_bytes=size)
        assert result.compared_bytes == size and result.mismatches == 1


def test_signed_zero_nan_payload_and_colliding_fold_patterns():
    source = torch.tensor([0, -2147483648, 2143289345, 2143289346], dtype=torch.int32).view(torch.float32)
    target = source.clone()
    scratch = torch.empty(7, dtype=torch.bool)
    assert compare_installed_views([(source, target)], scratch, expected_bytes=16).mismatches == 0
    target.view(torch.int32)[0] = -2147483648
    assert compare_installed_views([(source, target)], scratch, expected_bytes=16).mismatches == 1
    target = source.flip(0)
    assert compare_installed_views([(source, target)], scratch, expected_bytes=16).mismatches > 0
    target = source.clone()
    target.view(torch.uint8)[0] ^= 1
    target.view(torch.uint8)[8] ^= 1
    assert compare_installed_views([(source, target)], scratch, expected_bytes=16).mismatches == 2


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_actual_packed_expert_views_cover_round_robin_local_slots(projection):
    shape = (4, 3, 2) if projection != "down_proj" else (4, 2, 3)
    name = f"model.layers.0.mlp.experts.{projection}.weight"
    source = torch.arange(24, dtype=torch.bfloat16).reshape(shape)
    manifest = build_manifest([TensorSpec(name, shape, "bfloat16", True)], bucket_bytes=24)
    buffers = (torch.empty(24, dtype=torch.uint8), torch.empty(24, dtype=torch.uint8))
    w13, w2 = torch.zeros(2, 6, 2, dtype=torch.bfloat16), torch.zeros(2, 2, 3, dtype=torch.bfloat16)
    mapping = [-1, 0, -1, 1]
    for bucket in range(manifest.bucket_count):
        buffer = buffers[bucket % 2]
        pack_bucket(manifest, bucket, {name: source}, buffer)
        for entry, view in unpack_bucket(manifest, bucket, buffer):
            scatter_grug_experts(entry, view, w13, w2, mapping, "TRITON")
    scratch = torch.empty(5, dtype=torch.bool)
    compared = 0
    for bucket in range(manifest.bucket_count):
        buffer = buffers[bucket % 2]
        pack_bucket(manifest, bucket, {name: source}, buffer)
        for entry, view in unpack_bucket(manifest, bucket, buffer):
            pairs = grug_expert_views(entry, view, w13, w2, mapping, "TRITON")
            result = compare_installed_views(pairs, scratch, expected_bytes=12)
            assert result.mismatches == 0
            compared += result.compared_bytes
    assert compared == 24  # Two local experts, six bf16 elements each.


@pytest.mark.parametrize("expected", [3, 5])
def test_missing_or_extra_coverage_fails(expected):
    source = torch.arange(4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="coverage|cover"):
        compare_installed_views([(source, source.clone())], torch.empty(2, dtype=torch.bool), expected_bytes=expected)


def test_scratch_limit_and_empty_receiver_slice():
    scratch = torch.empty(MAX_SCRATCH_BYTES + 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="1 MiB"):
        compare_installed_views([], scratch, expected_bytes=0)
    result = compare_installed_views([], scratch[:1], expected_bytes=0)
    assert result.compared_bytes == result.mismatches == 0


def test_storage_coverage_rejects_equal_count_with_duplicated_or_missing_expert():
    installed = torch.empty(4, 6, dtype=torch.bfloat16)
    coverage = ReceiverByteCoverage({"experts": installed})
    for row in (3, 1, 0):
        coverage.observe(installed[row])
    with pytest.raises(ValueError, match="more than once"):
        coverage.observe(installed[1])
    with pytest.raises(ValueError, match="missed"):
        coverage.finish()
    coverage.observe(installed[2])
    assert coverage.finish() == 48


def test_storage_coverage_rejects_alias_inventory_and_unrelated_equal_tensor():
    installed = torch.zeros(8, dtype=torch.float32)
    with pytest.raises(ValueError, match="aliases"):
        ReceiverByteCoverage({"full": installed, "alias": installed[:4]})
    coverage = ReceiverByteCoverage({"full": installed})
    with pytest.raises(ValueError, match="outside"):
        coverage.observe(installed.clone())
    coverage.observe(installed)
    assert coverage.finish() == 32


def test_storage_coverage_detects_parameter_replacement():
    installed = torch.zeros(8, dtype=torch.float32)
    coverage = ReceiverByteCoverage({"full": installed})
    coverage.observe(installed)
    installed.data = installed.clone()
    with pytest.raises(ValueError, match="storage changed"):
        coverage.finish()


@pytest.mark.parametrize("alias", ["source", "installed", "self"])
def test_comparison_rejects_scratch_and_evidence_storage_overlap(alias):
    source = torch.arange(8, dtype=torch.uint8)
    installed = source.clone()
    scratch = torch.empty(3, dtype=torch.bool)
    if alias == "source":
        scratch = source[2:5].view(torch.bool)
    elif alias == "installed":
        scratch = installed[2:5].view(torch.bool)
    else:
        installed = source
    original = source.clone()
    with pytest.raises(ValueError, match="must not overlap"):
        compare_installed_views([(source, installed)], scratch, expected_bytes=8)
    assert torch.equal(source, original)
