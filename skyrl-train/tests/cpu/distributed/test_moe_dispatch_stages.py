from dataclasses import replace

import pytest

from tests.moe_dispatch_stages import (
    EXPECTED_DISPATCH_STAGES,
    DispatchStageRecord,
    dispatch_stage_records,
    validate_dispatch_stage_records,
)


def _complete_records() -> tuple[DispatchStageRecord, ...]:
    records = []
    for rank in range(4):
        for stage_index, stage in enumerate(EXPECTED_DISPATCH_STAGES):
            records.append(
                DispatchStageRecord(
                    rank=rank,
                    microbatch=0,
                    layer=0,
                    stage=stage,
                    ep_ranks=(0, 1, 2, 3),
                    sequence_number=stage_index // 2,
                    routed_rows=32,
                    input_splits=(8, 8, 8, 8),
                    output_splits=(8, 8, 8, 8),
                )
            )
    return tuple(records)


def test_dispatch_stage_records_round_trip_and_validate_complete_ep_group() -> None:
    expected = _complete_records()
    output = "\n".join(f"[worker] {record.json_line()}" for record in expected)

    actual = dispatch_stage_records(output)
    summary = validate_dispatch_stage_records(actual, world_size=4, microbatches=1, layers=1)

    assert actual == expected
    assert summary == {"records": 36, "ranks": 4, "microbatches": 1, "layers": 1}


def test_dispatch_stage_validation_identifies_first_rank_missing_after_token_enqueue() -> None:
    records = tuple(
        record
        for record in _complete_records()
        if not (record.rank == 2 and record.stage.value == "tokens_cuda_complete")
    )

    with pytest.raises(AssertionError, match="rank/microbatch/layer \\(2, 0, 0\\) stages"):
        validate_dispatch_stage_records(records, world_size=4, microbatches=1, layers=1)


def test_dispatch_stage_validation_rejects_ep_sequence_divergence() -> None:
    records = list(_complete_records())
    divergent_index = next(
        index
        for index, record in enumerate(records)
        if record.rank == 3 and record.stage.value == "tokens_a2a_before"
    )
    records[divergent_index] = replace(records[divergent_index], sequence_number=99)

    with pytest.raises(AssertionError, match="EP group .* sequence mismatch"):
        validate_dispatch_stage_records(records, world_size=4, microbatches=1, layers=1)
