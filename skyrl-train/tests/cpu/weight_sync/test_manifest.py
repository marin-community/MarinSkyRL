import json
import os
import subprocess
import sys

import pytest
import torch

from skyrl_train.weight_sync.manifest import TensorSpec, build_manifest, pack_bucket, unpack_bucket


def specs():
    return [
        TensorSpec("dense", (3, 2), "bfloat16"),
        TensorSpec("model.layers.0.mlp.experts.gate_proj.weight", (7, 2, 3), "bfloat16", True),
        TensorSpec("model.layers.0.mlp.router.bias", (7,), "float32"),
        TensorSpec("norm", (3,), "bfloat16"),
    ]


def test_buckets_never_exceed_threshold_and_keep_fp32_router_bias_separate():
    manifest = build_manifest(specs(), bucket_bytes=40)
    for bucket_id in range(manifest.bucket_count):
        entries = manifest.bucket(bucket_id)
        assert entries[-1].offset + entries[-1].nbytes <= 40
        assert entries[0].offset == 0
        for left, right in zip(entries, entries[1:]):
            assert left.offset + left.nbytes == right.offset
        if any(entry.wire_dtype == "float32" for entry in entries):
            assert len(entries) == 1
    expert = [e for e in manifest.entries if e.expert_start is not None]
    assert [(e.expert_start, e.shape[0]) for e in expert] == [(0, 2), (2, 3), (5, 2)]
    assert sum(e.numel for e in expert) == 7 * 2 * 3


def test_pack_unpack_roundtrip_bitwise():
    torch.manual_seed(7)
    source = {s.name: torch.randn(s.shape, dtype=getattr(torch, s.wire_dtype)) for s in specs()}
    # Preserve unusual IEEE bits, not merely equal numeric values.
    source["dense"].view(torch.int16).flatten()[0] = 32704
    reconstructed = {name: torch.empty_like(value) for name, value in source.items()}
    manifest = build_manifest(specs(), bucket_bytes=40)
    buffer = torch.empty(40, dtype=torch.uint8)
    for bucket_id in range(manifest.bucket_count):
        nbytes = pack_bucket(manifest, bucket_id, source, buffer)
        assert nbytes <= 40
        for entry, view in unpack_bucket(manifest, bucket_id, buffer):
            assert view.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
            reconstructed[entry.hf_name].flatten().narrow(0, entry.tensor_offset, entry.numel).copy_(view.flatten())
    for name in source:
        assert torch.equal(source[name].view(torch.uint8), reconstructed[name].view(torch.uint8)), name


def test_manifest_id_is_stable_across_processes():
    code = """from skyrl_train.weight_sync.manifest import TensorSpec,build_manifest
print(build_manifest([TensorSpec('x',(256,1280,2560),'bfloat16',True)]).manifest_id)
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code], text=True, env={**os.environ, "PYTHONHASHSEED": str(seed)}
        ).strip()
        for seed in (1, 77)
    ]
    assert outputs[0] == outputs[1]
    manifest = build_manifest([TensorSpec("x", (256, 1280, 2560), "bfloat16", True)])
    assert outputs[0] == manifest.manifest_id
    assert [(e.expert_start, e.shape[0]) for e in manifest.entries] == [(0, 163), (163, 93)]
    assert [e.nbytes for e in manifest.entries] == [1068236800, 609484800]


@pytest.mark.parametrize(
    "spec",
    [
        TensorSpec("x", (1,), "float16"),
        TensorSpec("x", (0,), "bfloat16"),
        TensorSpec("x", (100,), "bfloat16"),
        TensorSpec("x", (2, 3), "bfloat16", True),
    ],
)
def test_unsupported_shapes_and_unsplittable_tensors_fail(spec):
    with pytest.raises(ValueError):
        build_manifest([spec], bucket_bytes=40)


def test_pack_rejects_mismatched_source_before_mutating_buffer():
    source = {s.name: torch.zeros(s.shape, dtype=getattr(torch, s.wire_dtype)) for s in specs()}
    manifest = build_manifest(specs(), bucket_bytes=40)
    source[specs()[1].name] = source[specs()[1].name].float()
    buffer = torch.full((40,), 123, dtype=torch.uint8)
    with pytest.raises(ValueError, match="shape/dtype"):
        pack_bucket(manifest, 0, source, buffer)
    assert torch.equal(buffer, torch.full_like(buffer, 123))


def test_manifest_identity_includes_tensor_offsets_and_capacity():
    left, right = build_manifest(specs(), 40), build_manifest(specs(), 48)
    assert left.manifest_id != right.manifest_id
    assert len(left.manifest_id) == 64
    assert json.loads(json.dumps(left.manifest_id)) == left.manifest_id
