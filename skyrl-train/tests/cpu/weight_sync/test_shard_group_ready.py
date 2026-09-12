"""Warmup guards and failures; actual custom Gloo path lives in factory tests."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.shard_group_ready import warm_owned_groups


class Group:
    def rank(self):
        return 0

    def size(self):
        return 1


def runner():
    return SimpleNamespace(
        rank=0,
        completed=False,
        manifest_id=None,
        scratch=torch.zeros(16, dtype=torch.uint8),
        sources={"weight": torch.tensor([1, 2, 3], dtype=torch.int32)},
        parameters={},
    )


def test_warmup_uses_existing_workspace_and_checks_every_group(monkeypatch):
    instance = runner()
    pointer = instance.scratch.data_ptr()
    seen = []

    def broadcast(wire, *, src, group):
        assert wire.data_ptr() == pointer and wire.numel() == 1 and wire.dtype == torch.int32
        seen.append(wire.item())

    monkeypatch.setattr("skyrl_train.weight_sync.shard_group_ready.dist.broadcast", broadcast)
    result = warm_owned_groups(instance, {"first": Group(), "second": Group()}, {"first": (0,), "second": (0,)})
    assert seen == [1729, 1730]
    assert [row["readback"] for row in result["groups"]] == seen
    assert torch.equal(instance.sources["weight"], torch.tensor([1, 2, 3], dtype=torch.int32))


@pytest.mark.parametrize("invalid", ["alias", "missing", "wrong-rank", "already-installed"])
def test_warmup_rejects_invalid_preparation_before_transport(monkeypatch, invalid):
    instance = runner()
    group = Group()
    groups = {"group": group}
    if invalid == "alias":
        instance.sources["weight"] = instance.scratch.view(torch.int32)
    elif invalid == "missing":
        groups = {}
    elif invalid == "wrong-rank":
        group.rank = lambda: 1
    else:
        instance.completed = True

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid preparation reached a collective")

    monkeypatch.setattr("skyrl_train.weight_sync.shard_group_ready.dist.broadcast", forbidden)
    with pytest.raises(ValueError):
        warm_owned_groups(instance, groups, {"group": (0,)})


def test_native_warmup_failure_retains_exact_group_scope(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("native connection failed")

    monkeypatch.setattr("skyrl_train.weight_sync.shard_group_ready.dist.broadcast", fail)
    with pytest.raises(RuntimeError, match="native connection failed") as caught:
        warm_owned_groups(runner(), {"expert-0": Group()}, {"expert-0": (0,)})
    assert caught.value.__notes__ == ["Warmup group expert-0, local index 0, completed groups 0"]
