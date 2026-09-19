"""Verify an expert-block sync by replaying it (``generator.expert_block_sync.verify``).

Every root re-sends what it sent. Every receiver lands it in scratch, counts the
bytes that differ from what it installed, and checks that the bytes it compared
are all the parameter bytes it holds (a padded vocabulary tensor counts only its
HF rows). Every trainer rank also checks that its data-parallel peers hold
byte-identical parameters, since roots rotate across those peers.

The replay costs about one extra sync on the wire.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.expert_block.stream import Stream


@dataclass(frozen=True)
class ReplayReport:
    participant: int
    version: int
    compared_bytes: int
    parameter_bytes: int
    mismatched_bytes: int


def replay(stream: Stream, version: int) -> ReplayReport:
    """Re-run the sync's collectives; receivers compare each landed transfer against what they installed."""
    with torch.no_grad():
        return _replay(stream, version)


def _replay(stream: Stream, version: int) -> ReplayReport:
    landings = [(item, landing) for item, landing in stream.transfers() if landing is not None]
    scratch = torch.empty(
        max([landing.nbytes for _, landing in landings] + [0]), dtype=torch.uint8, device=stream.device
    )
    mismatched = torch.zeros((), dtype=torch.int64, device=stream.device)
    compared = 0
    for item, landing in stream.transfers():
        if landing is None:
            stream.send(item)
            continue
        wire = scratch.narrow(0, 0, landing.nbytes).view(landing.wire_dtype).view(landing.installed.shape)
        stream.receive(item, wire)
        # A widened parameter is compared at wire precision.
        installed = landing.installed.to(landing.wire_dtype)
        mismatched.add_(wire.view(torch.uint8).ne(installed.view(torch.uint8)).sum())
        compared += landing.nbytes
    parameter_bytes = 0
    if not stream.trainer:
        # Coverage counts every parameter byte at wire width, so a widened parameter counts as BF16.
        parameter_bytes = sum(value.numel() * value.element_size() for value in stream.parameters.values())
        for _, landing in landings:
            parameter_bytes -= landing.installed.numel() * (
                landing.installed.element_size() - landing.nbytes // landing.installed.numel()
            )
    if stream.device.type == "cuda":
        torch.cuda.synchronize(stream.device)
    return ReplayReport(stream.participant, version, compared, parameter_bytes, int(mismatched.item()))


@dataclass(frozen=True)
class ReplicaReport:
    participant: int
    version: int
    compared_bytes: int
    mismatched_bytes: int


# Bytes compared per collective. Each chunk is widened to int32 twice (the group minimum and
# maximum), so the working set is eight times this on a GPU the trainer already fills.
REPLICA_COMPARE_CHUNK_BYTES = 16 << 20


def compare_replicas(
    sources: dict[str, torch.Tensor],
    groups: dict[str, dist.ProcessGroup],
    participant: int,
    version: int,
    *,
    chunk_bytes: int = REPLICA_COMPARE_CHUNK_BYTES,
) -> ReplicaReport:
    """Count bytes on which this rank's parameters differ from the peers that hold the same ones.

    ``groups`` maps each parameter name to the process group of its replicas. Byte
    patterns are reduced as integers, so the comparison is exact.
    """
    mismatched = 0
    compared = 0
    with torch.no_grad():
        for name, source in sources.items():
            words = source.detach().contiguous().view(-1).view(torch.uint8)
            for start in range(0, words.numel(), chunk_bytes):
                low = words.narrow(0, start, min(chunk_bytes, words.numel() - start)).to(torch.int32)
                high = low.clone()
                dist.all_reduce(low, op=dist.ReduceOp.MIN, group=groups[name])
                dist.all_reduce(high, op=dist.ReduceOp.MAX, group=groups[name])
                mismatched += int(low.ne(high).sum().item())
            compared += words.numel()
    return ReplicaReport(participant, version, compared, mismatched)
