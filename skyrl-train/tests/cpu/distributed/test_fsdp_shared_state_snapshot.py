import torch

from skyrl_train.distributed.fsdp_strategy import snapshot_shared_state_dict_tensors


def test_snapshot_shared_state_dict_tensors_detaches_shared_storage_only():
    backing = torch.arange(8, dtype=torch.float32)
    tied = backing[:4].reshape(2, 2)
    distinct_view = backing[4:]
    unshared = torch.ones(2)
    empty_a, empty_b = torch.empty(0), torch.empty(0)
    state = {
        "embed.weight": tied,
        "lm_head.weight": tied.detach(),
        "view.weight": distinct_view,
        "norm.weight": unshared,
        "empty_a": empty_a,
        "empty_b": empty_b,
    }
    expected = {name: tensor.clone() for name, tensor in state.items()}

    snapshot = snapshot_shared_state_dict_tensors(state)

    assert snapshot["embed.weight"] is snapshot["lm_head.weight"]
    assert snapshot["norm.weight"] is unshared
    assert snapshot["empty_a"] is empty_a
    assert snapshot["empty_b"] is empty_b
    assert snapshot["view.weight"] is not distinct_view
    backing.zero_()
    for name in ("embed.weight", "lm_head.weight", "view.weight"):
        torch.testing.assert_close(snapshot[name], expected[name])
