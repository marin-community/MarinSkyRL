import pytest
import torch

from skyrl_train.weight_sync.bucket_receiver import GrugBucketReceiver
from skyrl_train.weight_sync.manifest import TensorSpec, build_manifest, pack_bucket


def fixture_parts():
    prefix = "model.layers.0.mlp.experts"
    specs = [TensorSpec("model.embed_tokens.weight", (4, 4), "bfloat16")]
    for projection in ("gate_proj", "up_proj", "down_proj"):
        specs.append(TensorSpec(f"{prefix}.{projection}.weight", (4, 2, 2), "bfloat16", True))
    specs.append(TensorSpec("model.layers.0.mlp.bias", (4,), "float32"))
    manifest = build_manifest(specs, bucket_bytes=32)
    sources = {
        spec.name: torch.arange(torch.tensor(spec.shape).prod().item(), dtype=getattr(torch, spec.wire_dtype))
        .reshape(spec.shape)
        .add(index * 100)
        for index, spec in enumerate(specs)
    }
    parameters = {
        "model.embed_tokens.weight": torch.full((4, 4), -1, dtype=torch.bfloat16),
        prefix + ".routed_experts.w13_weight": torch.full((2, 4, 2), -1, dtype=torch.bfloat16),
        prefix + ".routed_experts.w2_weight": torch.full((2, 2, 2), -1, dtype=torch.bfloat16),
        "model.layers.0.mlp.bias": torch.full((4,), -1, dtype=torch.float32),
    }
    # Nonconsecutive global experts, in a nonmonotonic receiver-local order.
    maps = {prefix + ".routed_experts": [-1, 1, -1, 0]}
    buffers = (torch.empty(32, dtype=torch.uint8), torch.empty(32, dtype=torch.uint8))
    return manifest, sources, parameters, maps, buffers


def receiver(parts):
    manifest, _, parameters, maps, buffers = parts
    return GrugBucketReceiver(manifest, parameters, maps, buffers, backend="TRITON", tensor_parallel_size=1)


def install(parts, session):
    manifest, sources, _, _, buffers = parts
    for bucket in range(manifest.bucket_count):
        pack_bucket(manifest, bucket, sources, buffers[bucket % 2])
        session.install_bucket(bucket)


def replay(parts, session):
    manifest, sources, _, _, buffers = parts
    scratch = torch.empty(7, dtype=torch.bool)
    for bucket in range(manifest.bucket_count):
        # Represents fresh frozen-sender transfer, not reuse of installed bytes.
        pack_bucket(manifest, bucket, sources, buffers[bucket % 2])
        session.replay_bucket(bucket, scratch)
    return session.finish_replay()


def test_native_layout_views_install_local_experts_and_compare_every_byte():
    parts = fixture_parts()
    session = receiver(parts)
    _, sources, parameters, _, _ = parts
    pointers = {name: tensor.data_ptr() for name, tensor in parameters.items()}
    assert all(torch.all(tensor == -1) for tensor in parameters.values())
    install(parts, session)
    prefix = "model.layers.0.mlp.experts"
    w13 = parameters[prefix + ".routed_experts.w13_weight"]
    assert torch.equal(w13[0, :2], sources[prefix + ".gate_proj.weight"][3])
    assert torch.equal(w13[1, 2:], sources[prefix + ".up_proj.weight"][1])
    result = replay(parts, session)
    assert result.mismatches == 0
    assert result.compared_bytes == sum(t.numel() * t.element_size() for t in parameters.values())
    assert pointers == {name: tensor.data_ptr() for name, tensor in parameters.items()}


@pytest.mark.parametrize("target", ["model.embed_tokens.weight", "model.layers.0.mlp.bias"])
def test_exact_replay_detects_one_changed_installed_byte(target):
    parts = fixture_parts()
    session = receiver(parts)
    install(parts, session)
    parts[2][target].view(torch.uint8).view(-1)[-1] ^= 1
    assert replay(parts, session).mismatches == 1


def test_unrepresented_installed_parameter_rejects_before_any_copy():
    parts = fixture_parts()
    parts[2]["unexpected.weight"] = torch.ones(3, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="missed installed receiver bytes"):
        receiver(parts)
    assert torch.all(parts[2]["model.embed_tokens.weight"] == -1)


def test_duplicate_transfer_buffer_rejects_before_any_copy():
    parts = fixture_parts()
    parts = (*parts[:4], (parts[4][0], parts[4][0]))
    with pytest.raises(ValueError, match="must not overlap"):
        receiver(parts)


def test_padded_dense_layout_rejected_before_any_copy():
    parts = fixture_parts()
    parts[2]["model.embed_tokens.weight"] = torch.empty(5, 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="match the complete wire tensor"):
        receiver(parts)


def test_replay_rejects_skipped_and_repeated_buckets():
    parts = fixture_parts()
    session = receiver(parts)
    with pytest.raises(ValueError, match="after every install"):
        session.replay_bucket(0, torch.empty(7, dtype=torch.bool))
    with pytest.raises(ValueError, match="manifest order"):
        session.install_bucket(1)
    install(parts, session)
    with pytest.raises(ValueError, match="missing manifest buckets"):
        session.finish_replay()
    pack_bucket(parts[0], 0, parts[1], parts[4][0])
    session.replay_bucket(0, torch.empty(7, dtype=torch.bool))
    with pytest.raises(ValueError, match="manifest order"):
        session.replay_bucket(0, torch.empty(7, dtype=torch.bool))


def test_nontriton_or_tensor_sharded_receiver_rejected():
    manifest, _, parameters, maps, buffers = fixture_parts()
    for backend, tp in (("FLASHINFER", 1), ("TRITON", 2)):
        with pytest.raises(ValueError, match="TP1 TRITON"):
            GrugBucketReceiver(manifest, parameters, maps, buffers, backend=backend, tensor_parallel_size=tp)


def test_replay_scratch_cannot_corrupt_the_other_transfer_slot():
    parts = fixture_parts()
    session = receiver(parts)
    install(parts, session)
    pack_bucket(parts[0], 0, parts[1], parts[4][0])
    other = parts[4][1]
    original = other.clone()
    with pytest.raises(ValueError, match="overlaps transfer"):
        session.replay_bucket(0, other.view(torch.bool))
    assert torch.equal(other, original)
