"""Triton 3.6 raw-bit kernels for the standalone sparse replay benchmark."""

import triton
import triton.language as tl


@triton.jit
def count_changes(base, target, counts, elements, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    changed = (offsets < elements) & (
        tl.load(base + offsets, offsets < elements, 0) != tl.load(target + offsets, offsets < elements, 0)
    )
    tl.store(counts + block, tl.sum(changed.to(tl.int32), 0))


@triton.jit
def pack_changes(base, target, offsets, indices, values, elements, LOCAL: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    positions = block * BLOCK + local
    after = tl.load(target + positions, positions < elements, 0)
    changed = (positions < elements) & (tl.load(base + positions, positions < elements, 0) != after)
    packed = tl.load(offsets + block) + tl.cumsum(changed.to(tl.int32), 0) - 1
    tl.store(indices + packed, local if LOCAL else positions, changed)
    tl.store(values + packed, after, changed)


@triton.jit
def validate_indices(indices, offsets, errors, elements, entries, LOCAL: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    if LOCAL:
        begin, end = tl.load(offsets + block), tl.load(offsets + block + 1)
        blocks = tl.cdiv(elements, BLOCK)
        valid = (begin >= 0) & (end >= begin) & (end <= entries) & (end - begin <= BLOCK)
        valid = valid & ((block != 0) | (begin == 0)) & ((block != blocks - 1) | (end == entries))
        positions = begin + lane
        mask = valid & (lane < end - begin)
        index = tl.load(indices + positions, mask, 0).to(tl.int32)
        previous = tl.load(indices + positions - 1, mask & (lane > 0), -1).to(tl.int32)
        bad = mask & ((index < 0) | (index >= BLOCK) | (block * BLOCK + index >= elements) | (index <= previous))
        tl.store(errors + block, tl.sum(bad.to(tl.int32), 0) + (~valid).to(tl.int32))
    else:
        positions = block * BLOCK + lane
        mask = positions < entries
        index = tl.load(indices + positions, mask, 0)
        previous = tl.load(indices + positions - 1, mask & (positions > 0), -1)
        bad = mask & ((index < 0) | (index >= elements) | (index <= previous))
        tl.store(errors + block, tl.sum(bad.to(tl.int32), 0))


@triton.jit
def apply_changes(indices, offsets, values, staged, entries, LOCAL: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    if LOCAL:
        begin, end = tl.load(offsets + block), tl.load(offsets + block + 1)
        positions = begin + lane
        mask = positions < end
        index = block * BLOCK + tl.load(indices + positions, mask, 0).to(tl.int32)
    else:
        positions = block * BLOCK + lane
        mask = positions < entries
        index = tl.load(indices + positions, mask, 0)
    value = tl.load(values + positions, mask, 0)
    tl.store(staged + index, value, mask)
