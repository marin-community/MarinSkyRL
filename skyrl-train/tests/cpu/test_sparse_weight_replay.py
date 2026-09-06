"""Independent byte oracle and corruption cases for the unused transport prototype."""

from dataclasses import replace
import hashlib
import struct

import pytest

from skyrl_train.entrypoints.benchmark_sparse_weight_replay import cpu_replay, make_target
from skyrl_train.sparse_weight_replay import (
    HostBaselineCache,
    MAX_CACHE_BYTES,
    MAX_CHUNK_BYTES,
    decode_chunk,
    encode_chunk,
)


@pytest.mark.parametrize("dtype,width", [("bfloat16", 2), ("float32", 4)])
@pytest.mark.parametrize("fraction", [0, 0.02, 0.03, 0.2, 1])
@pytest.mark.parametrize("elements", [0, 1, 1031])
def test_exact_independent_replacement_oracle(dtype, width, fraction, elements):
    # Arbitrary raw patterns include signed zero, NaN payloads and infinities;
    # float equality or subtract/add reconstruction is not a valid oracle.
    patterns = [0, 1 << (8 * width - 1), (1 << (8 * width)) - 1, 0x7F80 if width == 2 else 0x7F800000]
    base = b"".join(patterns[i % len(patterns)].to_bytes(width, "little") for i in range(elements))
    changed = set(range(round(elements * fraction)))
    target = b"".join(
        (int.from_bytes(base[i * width : (i + 1) * width], "little") ^ (1 if i in changed else 0)).to_bytes(
            width, "little"
        )
        for i in range(elements)
    )
    for mode in ("dense", "indexed", "auto"):
        packet, stats = encode_chunk(base, target, dtype=dtype, base_version=2, target_version=4, mode=mode)
        reconstructed, _ = decode_chunk(base, packet, installed_version=2)
        assert reconstructed == target
        assert stats["changed_elements"] == len(changed)
        assert packet.target_sha256 == hashlib.sha256(target).hexdigest()
        if packet.encoding == "indexed":
            actual_indices = [
                int.from_bytes(packet.indices[i : i + 4], "little") for i in range(0, len(packet.indices), 4)
            ]
            assert actual_indices == sorted(changed)
            assert packet.values == b"".join(target[i * width : (i + 1) * width] for i in sorted(changed))
        if mode == "auto":
            assert packet.payload_bytes <= len(target)


@pytest.fixture
def patch_example():
    base = struct.pack("<4H", 0, 1, 2, 3)
    target = struct.pack("<4H", 0, 9, 2, 8)
    packet, _ = encode_chunk(base, target, dtype="bfloat16", base_version=0, target_version=1, mode="indexed")
    return base, packet


@pytest.mark.parametrize("indices", [[1, 1], [3, 1], [1, 4], [0xFFFFFFFF, 1]])
def test_duplicate_unsorted_and_out_of_range_indices_rejected(patch_example, indices):
    base, packet = patch_example
    bad = replace(packet, indices=struct.pack("<2I", *indices))
    with pytest.raises(ValueError, match="Indices"):
        decode_chunk(base, bad, installed_version=0)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("indices", b"x", "lengths"),
        ("values", b"x", "lengths"),
        ("encoding", "xor", "Unknown"),
        ("base_sha256", "0" * 64, "Base checksum"),
        ("target_sha256", "0" * 64, "Target checksum"),
        ("elements", 5, "element count"),
        ("dtype", "float16", "Only raw"),
        ("target_version", 0, "strictly newer"),
    ],
)
def test_malformed_patch_never_changes_baseline(patch_example, field, value, error):
    base, packet = patch_example
    before = bytes(base)
    with pytest.raises(ValueError, match=error):
        decode_chunk(base, replace(packet, **{field: value}), installed_version=0)
    assert base == before


def test_wrong_installed_version_or_corrupted_baseline_rejected(patch_example):
    base, packet = patch_example
    with pytest.raises(ValueError, match="Installed base"):
        decode_chunk(base, packet, installed_version=1)
    with pytest.raises(ValueError, match="Base checksum"):
        decode_chunk(b"xx" + base[2:], packet, installed_version=0)


def test_dense_payload_rejects_indices_and_truncation(patch_example):
    base, _ = patch_example
    packet, _ = encode_chunk(base, base, dtype="bfloat16", base_version=0, target_version=1, mode="dense")
    for bad in (replace(packet, indices=b"xxxx"), replace(packet, values=base[:-2])):
        with pytest.raises(ValueError, match="Malformed dense"):
            decode_chunk(base, bad, installed_version=0)


def test_bounds_alignment_and_versions_fail_before_encoding():
    for base, target in [(b"x", b"x"), (b"xx", b"xxxx"), (bytes(MAX_CHUNK_BYTES + 2), b"xx")]:
        with pytest.raises(ValueError):
            encode_chunk(base, target, dtype="bfloat16", base_version=0, target_version=1)
    for version in (-1, True, 0.5):
        with pytest.raises(ValueError, match="version"):
            encode_chunk(b"xx", b"yy", dtype="bfloat16", base_version=version, target_version=1)


def test_cache_admission_is_bounded_without_eviction_or_partial_mutation():
    cache = HostBaselineCache(max_bytes=8)
    cache.add("first", b"abcd")
    cache.add("tail", b"efgh")
    for key, value in [("overflow", b"x"), ("first", b"xx")]:
        with pytest.raises(ValueError):
            cache.add(key, value)
        assert cache.retained_bytes == 8
        assert cache.get("first") + cache.get("tail") == b"abcdefgh"
    with pytest.raises(ValueError):
        HostBaselineCache(MAX_CACHE_BYTES + 1)


def test_replay_exercises_partial_tail_and_reports_actual_payload_population():
    cache = HostBaselineCache(max_bytes=40)
    cache.add("whole", bytes(range(32)))
    cache.add("tail", b"abcdef")
    for mode in ("dense", "indexed"):
        result = cpu_replay(cache, ["whole", "tail"], "bfloat16", 1, mode)
        assert result["changed_elements"] == 19
        assert result["payload_bytes"] == (38 if mode == "dense" else 19 * 6)
    assert make_target(bytes(range(32)), "bfloat16", 0, 17) == bytes(range(32))


def test_bit_changed_nan_and_negative_zero_are_not_silently_equal():
    base = struct.pack("<4I", 0, 0x7FC00001, 0x7F800000, 0xFF800000)
    target = struct.pack("<4I", 0x80000000, 0x7FC00002, 0x7F800000, 0xFF800000)
    packet, stats = encode_chunk(base, target, dtype="float32", base_version=0, target_version=2, mode="indexed")
    assert stats["changed_elements"] == 2
    assert packet.indices == struct.pack("<2I", 0, 1)
    assert decode_chunk(base, packet, installed_version=0)[0] == target
