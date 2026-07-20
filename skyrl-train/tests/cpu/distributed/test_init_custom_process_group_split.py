"""Regression guard for init_custom_process_group's cross-world weight-sync group creation.

The bug it prevents (reproduced on cw-us-east-02a AND cw-rno2a, 2026-07-20): the disaggregated
trainer->engine weight-sync NCCL group (group_name "skyrl", world_size = megatron rank 0 + all
vLLM engine ranks) hung at its first bootstrap. Every engine rank blocked in
``broadcastUniqueNCCLID`` doing ``store->get('0')`` and timed out after 1800000ms because rank 0
(the trainer) never did the matching ``store->set('0')``.

Root cause: ``_new_process_group_helper`` auto-splits a new NCCL comm from the caller's default
process group whenever that default group is device-bound (``bound_device_id`` set). The megatron
policy worker pins its default PG with a device_id, so on the trainer the weight-sync comm was
built by ``ncclCommSplit`` from the trainer-only default group (excluding the engines), while the
engines built it via a plain store-based ``ncclCommInitRank``. The two paths are incompatible for
this cross-world group, so no rank published the uniqueId and the group deadlocked.

The fix: ``init_custom_process_group`` un-binds the default group's device for the duration of the
``_new_process_group_helper`` call, forcing ``split_from = None`` so every rank takes the identical
fresh store-uniqueId path. It must restore the default group's ``bound_device_id`` afterward.
"""

import torch

import skyrl_train.distributed.utils as dist_utils


class _FakeDefaultPG:
    """Mimics a device-pinned default process group (bound_device_id set)."""

    def __init__(self, bound_device_id):
        self.bound_device_id = bound_device_id


class _FakeStore:
    """Placeholder store so init_custom_process_group skips rendezvous."""


def _install_fakes(monkeypatch, default_pg):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(dist_utils, "_get_default_group", lambda: default_pg)


def test_disables_split_from_device_bound_default_group(monkeypatch):
    """With a device-bound default group, the helper must see bound_device_id=None during the
    call (split disabled), and the default group must be restored afterward."""
    bound = torch.device("cuda", 0)
    default_pg = _FakeDefaultPG(bound)
    _install_fakes(monkeypatch, default_pg)

    seen = {}

    def _fake_helper(world_size, rank, ranks, backend, store, group_name, backend_options=None, timeout=None):
        # Captured at the exact moment the NCCL backend / comm-split decision is made.
        seen["bound_during_call"] = default_pg.bound_device_id
        seen["ranks"] = ranks
        return object(), None

    monkeypatch.setattr(dist_utils, "_new_process_group_helper", _fake_helper)

    pg = dist_utils.init_custom_process_group(
        backend="nccl", store=_FakeStore(), world_size=17, rank=0, group_name="skyrl"
    )

    # Split-from-default was disabled while the comm was created ...
    assert seen["bound_during_call"] is None
    # ... it is still created as a standalone group ([] global ranks) ...
    assert seen["ranks"] == []
    # ... the default group's device binding is restored (nothing else about it changes) ...
    assert default_pg.bound_device_id == bound
    # ... and the group's global-rank map spans the full world.
    assert dist_utils._world.pg_group_ranks[pg] == {i: i for i in range(17)}
    dist_utils._world.pg_group_ranks.pop(pg, None)


def test_restores_default_binding_even_if_helper_raises(monkeypatch):
    bound = torch.device("cuda", 1)
    default_pg = _FakeDefaultPG(bound)
    _install_fakes(monkeypatch, default_pg)

    def _boom(*args, **kwargs):
        assert default_pg.bound_device_id is None  # disabled inside the call
        raise RuntimeError("helper failed")

    monkeypatch.setattr(dist_utils, "_new_process_group_helper", _boom)

    try:
        dist_utils.init_custom_process_group(
            backend="nccl", store=_FakeStore(), world_size=17, rank=0, group_name="skyrl"
        )
    except RuntimeError:
        pass

    # The finally-block restores the binding regardless of failure.
    assert default_pg.bound_device_id == bound


def test_noop_when_default_group_not_device_bound(monkeypatch):
    """Engine side (default group not device-bound): nothing is mutated, fresh path is used."""
    default_pg = _FakeDefaultPG(None)
    _install_fakes(monkeypatch, default_pg)

    seen = {}

    def _fake_helper(world_size, rank, ranks, backend, store, group_name, backend_options=None, timeout=None):
        seen["bound_during_call"] = default_pg.bound_device_id
        return object(), None

    monkeypatch.setattr(dist_utils, "_new_process_group_helper", _fake_helper)

    pg = dist_utils.init_custom_process_group(
        backend="nccl", store=_FakeStore(), world_size=17, rank=5, group_name="skyrl"
    )

    assert seen["bound_during_call"] is None
    assert default_pg.bound_device_id is None
    dist_utils._world.pg_group_ranks.pop(pg, None)
