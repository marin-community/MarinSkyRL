import torch

from skyrl_train.distributed.fsdp_strategy import snapshot_shared_state_dict_tensors


def test_snapshot_shared_state_dict_tensors_detaches_tied_storage_only():
    tied = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    unshared = torch.ones(2)
    state = {
        "embed.weight": tied,
        "lm_head.weight": tied.detach(),
        "norm.weight": unshared,
    }

    snapshot = snapshot_shared_state_dict_tensors(state)

    assert snapshot["embed.weight"] is snapshot["lm_head.weight"]
    assert snapshot["embed.weight"].untyped_storage().data_ptr() != tied.untyped_storage().data_ptr()
    assert snapshot["norm.weight"] is unshared
    torch.testing.assert_close(snapshot["embed.weight"], tied)

    tied.zero_()
    torch.testing.assert_close(snapshot["embed.weight"], torch.arange(12, dtype=torch.float32).reshape(3, 4))


def test_snapshot_shared_state_dict_tensors_does_not_group_empty_storage():
    first = torch.empty(0)
    second = torch.empty(0)

    snapshot = snapshot_shared_state_dict_tensors({"first": first, "second": second})

    assert snapshot["first"] is first
    assert snapshot["second"] is second


def test_snapshot_shared_state_dict_tensors_preserves_distinct_views():
    backing = torch.arange(8, dtype=torch.float32)
    left = backing[:4]
    right = backing[4:]

    snapshot = snapshot_shared_state_dict_tensors({"left": left, "right": right})

    torch.testing.assert_close(snapshot["left"], torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(snapshot["right"], torch.arange(4, 8, dtype=torch.float32))
    assert snapshot["left"] is not snapshot["right"]
    backing.zero_()
    torch.testing.assert_close(snapshot["left"], torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(snapshot["right"], torch.arange(4, 8, dtype=torch.float32))
