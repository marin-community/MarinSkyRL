"""Structured records and validation for the MoE dispatch discriminator."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum


DISPATCH_STAGE_MARKER = "MOE_DISPATCH_STAGE"
DISPATCH_MICROBATCHES = 8
DISPATCH_LAYERS = 3


class DispatchStage(StrEnum):
    MOE_ENTER = "moe_enter"
    ROUTING_COMPLETE = "routing_complete"
    COUNTS_A2A_BEFORE = "counts_a2a_before"
    COUNTS_A2A_AFTER = "counts_a2a_after"
    SPLITS_CONSTRUCTED = "splits_constructed"
    TOKENS_A2A_BEFORE = "tokens_a2a_before"
    TOKENS_A2A_AFTER = "tokens_a2a_after"
    TOKENS_CUDA_COMPLETE = "tokens_cuda_complete"
    MOE_EXIT = "moe_exit"


EXPECTED_DISPATCH_STAGES = tuple(DispatchStage)


@dataclass(frozen=True)
class DispatchStageRecord:
    rank: int
    microbatch: int
    layer: int
    stage: DispatchStage
    ep_ranks: tuple[int, ...]
    sequence_number: int
    routed_rows: int
    input_splits: tuple[int, ...] = ()
    output_splits: tuple[int, ...] = ()

    def json_line(self) -> str:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        return f"{DISPATCH_STAGE_MARKER} {json.dumps(payload, sort_keys=True)}"


def dispatch_stage_records(output: str) -> tuple[DispatchStageRecord, ...]:
    """Parse dispatch records from torchrun output, tolerating launcher prefixes."""

    records = []
    for line in output.splitlines():
        marker_index = line.find(f"{DISPATCH_STAGE_MARKER} ")
        if marker_index < 0:
            continue
        payload = json.loads(line[marker_index + len(DISPATCH_STAGE_MARKER) + 1 :])
        records.append(
            DispatchStageRecord(
                rank=int(payload["rank"]),
                microbatch=int(payload["microbatch"]),
                layer=int(payload["layer"]),
                stage=DispatchStage(payload["stage"]),
                ep_ranks=tuple(int(rank) for rank in payload["ep_ranks"]),
                sequence_number=int(payload["sequence_number"]),
                routed_rows=int(payload["routed_rows"]),
                input_splits=tuple(int(value) for value in payload["input_splits"]),
                output_splits=tuple(int(value) for value in payload["output_splits"]),
            )
        )
    return tuple(records)


def validate_dispatch_stage_records(
    records: Iterable[DispatchStageRecord],
    *,
    world_size: int,
    microbatches: int,
    layers: int,
) -> dict[str, int]:
    """Require complete ordered dispatch stages and matching EP-group sequences."""

    records = tuple(records)
    grouped: dict[tuple[int, int, int], list[DispatchStageRecord]] = {}
    for record in records:
        grouped.setdefault((record.rank, record.microbatch, record.layer), []).append(record)

    expected_keys = {
        (rank, microbatch, layer)
        for rank in range(world_size)
        for microbatch in range(microbatches)
        for layer in range(layers)
    }
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        unexpected = sorted(set(grouped) - expected_keys)
        raise AssertionError(f"dispatch record keys differ: missing={missing[:8]} unexpected={unexpected[:8]}")

    for key, local_records in grouped.items():
        stages = tuple(record.stage for record in local_records)
        if stages != EXPECTED_DISPATCH_STAGES:
            raise AssertionError(f"rank/microbatch/layer {key} stages {stages} != {EXPECTED_DISPATCH_STAGES}")

    for microbatch in range(microbatches):
        for layer in range(layers):
            ep_groups: dict[tuple[int, ...], list[list[DispatchStageRecord]]] = {}
            for rank in range(world_size):
                local_records = grouped[(rank, microbatch, layer)]
                ep_groups.setdefault(local_records[0].ep_ranks, []).append(local_records)
            for ep_ranks, member_records in ep_groups.items():
                if len(member_records) != len(ep_ranks):
                    raise AssertionError(
                        f"EP group {ep_ranks} has {len(member_records)} rank records; expected {len(ep_ranks)}"
                    )
                reference = tuple(record.sequence_number for record in member_records[0])
                for local_records in member_records[1:]:
                    candidate = tuple(record.sequence_number for record in local_records)
                    if candidate != reference:
                        raise AssertionError(
                            f"EP group {ep_ranks} sequence mismatch at microbatch={microbatch} layer={layer}: "
                            f"{reference} != {candidate}"
                        )

    return {
        "records": len(records),
        "ranks": world_size,
        "microbatches": microbatches,
        "layers": layers,
    }
