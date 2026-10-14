import warnings

import pytest
import torch
from torch.distributed import checkpoint

from skyrl_train.io.checkpoint_buffers import merge_adjacent_checkpoint_shards


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("offset", [0, 17])
def test_checkpoint_restore_joins_gate_and_up_without_another_buffer(tmp_path, dtype, offset):
    expected = torch.arange(120, dtype=dtype).reshape(12, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        checkpoint.save({"gate": expected[:6], "up": expected[6:]}, checkpoint_id=tmp_path)

    allocation = torch.full((offset + expected.numel() + 7,), -1, dtype=dtype)
    destination = allocation[offset : offset + expected.numel()].view_as(expected)
    gate, up = destination.chunk(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        checkpoint.load({"gate": gate, "up": up}, checkpoint_id=tmp_path)
    restored = merge_adjacent_checkpoint_shards([gate, up], torch.cat)

    assert torch.equal(restored, expected)
    assert restored.data_ptr() == destination.data_ptr(), "restore must reuse the existing full destination"
    assert torch.equal(allocation[:offset], torch.full((offset,), -1, dtype=dtype))
    assert torch.equal(allocation[-7:], torch.full((7,), -1, dtype=dtype))


@pytest.mark.parametrize("layout", ["separate", "reversed", "gapped", "transposed"])
def test_checkpoint_merge_preserves_values_for_other_shard_layouts(layout):
    source = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    if layout == "separate":
        shards = [source[:4].clone(), source[4:].clone()]
    elif layout == "reversed":
        shards = [source[4:], source[:4]]
    elif layout == "gapped":
        shards = [source[:3], source[4:]]
    else:
        shards = [source[:4].T, source[4:].T]

    expected = torch.cat(shards)
    restored = merge_adjacent_checkpoint_shards(shards, torch.cat)
    assert torch.equal(restored, expected)
