"""Exact source ownership and byte coverage for bounded frozen replay."""

from dataclasses import dataclass
import math

from skyrl_train.weight_sync.frozen_source_views import FrozenSourceSlice


@dataclass(frozen=True)
class BucketSourceSlice:
    owner_rank: int
    source: FrozenSourceSlice
    bucket_offset: int


def frozen_source_plan(manifest, gathered):
    if sorted(row["rank"] for row in gathered) != list(range(len(gathered))):
        raise ValueError("Frozen source catalogue is missing policy ranks")
    by_name = {}
    for row in gathered:
        for item in row["slices"]:
            source = FrozenSourceSlice(**item)
            if not row["expert_owner" if source.expert else "dense_owner"]:
                continue
            by_name.setdefault(source.hf_name, []).append((row["rank"], source))
    expected = {entry.hf_name: entry for entry in manifest.entries}
    if set(by_name) != set(expected):
        raise ValueError("Frozen source names do not cover the complete HF manifest")
    for name, values in by_name.items():
        values.sort(key=lambda pair: pair[1].hf_offset)
        cursor = 0
        for _, source in values:
            if source.hf_offset != cursor or source.numel <= 0 or source.wire_dtype != expected[name].wire_dtype:
                raise ValueError("Frozen source ownership has an overlap, gap or dtype mismatch")
            cursor += source.numel
        if cursor != math.prod(expected[name].full_shape):
            raise ValueError("Frozen source ownership misses HF tensor elements")
    plan = []
    for bucket in range(manifest.bucket_count):
        parts = []
        for entry in manifest.bucket(bucket):
            width = entry.nbytes // entry.numel
            for owner, source in by_name[entry.hf_name]:
                begin = max(entry.tensor_offset, source.hf_offset)
                end = min(entry.tensor_offset + entry.numel, source.hf_offset + source.numel)
                if begin >= end:
                    continue
                section = FrozenSourceSlice(
                    source.hf_name,
                    begin,
                    end - begin,
                    source.wire_dtype,
                    source.source_key,
                    source.source_offset + begin - source.hf_offset,
                    source.expert,
                )
                parts.append(BucketSourceSlice(owner, section, entry.offset + (begin - entry.tensor_offset) * width))
        cursor = 0
        for part in parts:
            if part.bucket_offset != cursor:
                raise ValueError("Frozen replay schedule leaves a bucket gap")
            width = 2 if part.source.wire_dtype == "bfloat16" else 4
            cursor += part.source.numel * width
        entry = manifest.bucket(bucket)[-1]
        if cursor != entry.offset + entry.nbytes:
            raise ValueError("Frozen replay schedule does not fill the complete bucket")
        plan.append(tuple(parts))
    return tuple(plan)
