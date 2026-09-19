"""Raw-bit behavior of the disposable routed sparse codecs."""

import pytest
import torch

from skyrl_train.weight_sync.expert_block.sparse_experiment_codec import apply, bits, encode


@pytest.mark.parametrize("encoding", ["indices", "bitmap"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sparse_patch_preserves_signed_zero_and_nan_payloads(encoding, dtype):
    integer = torch.int16 if dtype == torch.bfloat16 else torch.int32
    raw_before = [0x0000, 0x7F80, 0x7FC1, 0x3F80, 0x8000, 0x7FC2, 0x0000, 0x3F80, 0x7F81]
    raw_after = [0x8000, 0x7F81, 0x7FC2, 0x3F80, 0x0000, 0x7FC2, 0x0000, 0x3F80, 0x7FC2]
    if dtype == torch.float32:
        raw_before = [value << 16 for value in raw_before]
        raw_after = [value << 16 for value in raw_after]
    baseline = torch.tensor(raw_before, dtype=torch.int64).to(integer).view(dtype)
    current = torch.tensor(raw_after, dtype=torch.int64).to(integer).view(dtype)
    patch = encode(current, baseline, encoding)
    installed = baseline.clone()
    apply(installed, patch)
    assert torch.equal(bits(installed), bits(current))
    assert patch.changed == 5


@pytest.mark.parametrize("encoding", ["indices", "bitmap"])
def test_sparse_patch_matches_dense_widened_router(encoding):
    before = torch.tensor([0, -32768, 16256, 16384, 1, 2, 3, 4, 5], dtype=torch.int16).view(torch.bfloat16)
    after = torch.tensor([-32768, 0, 16256, 16385, 1, 2, 3, 4, 32641], dtype=torch.int16).view(torch.bfloat16)
    installed = before.float()
    apply(installed, encode(after, before, encoding))
    assert torch.equal(bits(installed), bits(after.float()))


@pytest.mark.parametrize("encoding", ["indices", "bitmap"])
def test_sparse_patch_handles_no_changes_and_non_byte_aligned_length(encoding):
    original = torch.arange(11, dtype=torch.float32).to(torch.bfloat16)
    patch = encode(original, original.clone(), encoding)
    installed = original.clone()
    apply(installed, patch)
    assert patch.changed == 0
    assert torch.equal(bits(installed), bits(original))
