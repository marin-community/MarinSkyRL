"""Force custom communicator connection during measured one-time preparation."""

import time

import torch
import torch.distributed as dist


def warm_owned_groups(runner, groups, memberships):
    """Use only the validated transfer buffer, before binding a weight sync.

    Custom NCCL groups may connect lazily at their first collective. This
    acknowledged payload keeps that work outside later weight sync timers.
    Source and installed parameter bytes are never used as warmup storage.
    """
    if runner.completed or runner.manifest_id is not None:
        raise ValueError("Group warmup belongs to an unused prepared runner")
    scratch = runner.scratch
    if scratch.dtype != torch.uint8 or scratch.ndim != 1 or not scratch.is_contiguous() or scratch.numel() < 4:
        raise ValueError("Group warmup requires the existing byte transfer workspace")
    left, right = scratch.data_ptr(), scratch.data_ptr() + scratch.numel()
    for tensor in (*runner.sources.values(), *runner.parameters.values()):
        if tensor.device != scratch.device or max(left, tensor.data_ptr()) < min(
            right, tensor.data_ptr() + tensor.numel() * tensor.element_size()
        ):
            raise ValueError("Group warmup must not overlap live source or installed storage")
    expected = tuple(name for name, members in memberships.items() if runner.rank in members)
    if tuple(groups) != expected:
        raise ValueError("Group warmup requires every owned group in deterministic order")
    for name, group in groups.items():
        members = memberships[name]
        if group.rank() != members.index(runner.rank) or group.size() != len(members):
            raise ValueError("Warmup group identity differs from prepared membership")
    if scratch.is_cuda:
        torch.cuda.synchronize(scratch.device)
    started = time.monotonic()
    wire = scratch.narrow(0, 0, 4).view(torch.int32)
    rows = []
    for index, (name, group) in enumerate(groups.items()):
        group_started = time.monotonic()
        expected_value = 1729 + list(memberships).index(name)
        wire.fill_(expected_value if group.rank() == 0 else -1)
        try:
            dist.broadcast(wire, src=0, group=group)
            # .item() waits for the actual rank-local CUDA result too.
            if wire.item() != expected_value:
                raise ValueError("Group warmup payload readback differs")
        except BaseException as error:
            error.add_note(f"Warmup group {name}, local index {index}, completed groups {len(rows)}")
            raise
        rows.append(
            {
                "name": name,
                "members": memberships[name],
                "root_global_rank": memberships[name][0],
                "payload_bytes": 4,
                "readback": expected_value,
                "seconds": time.monotonic() - group_started,
            }
        )
    return {
        "phase": "groups-ready",
        "seconds": time.monotonic() - started,
        "groups": rows,
        "cuda_measured": scratch.is_cuda,
        "new_explicit_tensor_storage_bytes": 0,
        "existing_transfer_workspace_bytes": scratch.numel(),
        "scope": "One-time connection/warmup; payload arithmetic is not physical NIC traffic",
    }
