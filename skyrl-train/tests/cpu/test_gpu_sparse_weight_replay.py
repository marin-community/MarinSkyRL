"""CPU transaction gates for the opt-in GPU replay; no CUDA dependency in CI."""

from dataclasses import replace
import struct

import pytest
import torch

from skyrl_train.gpu_sparse_weight_replay import Encoding, ReplayBaseline, packet_header


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("mode", [Encoding.INDEX32, Encoding.BLOCK_LOCAL16])
def test_sparse_payload_never_expands_dense_and_empty_patch_advances_version(width, mode):
    for changed in (0, 1, 50, 512, 1025):
        header = packet_header(
            elements=1025, width=width, changed=changed, preferred=mode, base_version=2, target_version=7
        )
        assert header.payload_bytes <= 1025 * width
        if changed == 0:
            assert header.encoding == Encoding.NOOP
            state = ReplayBaseline(torch.zeros(1025, dtype=torch.int16 if width == 2 else torch.int32), version=2)
            previous = state.values
            state.install(header, previous.clone(), previous.clone())
            assert state.version == 7 and torch.equal(state.values, previous)
        if changed == 1025:
            assert header.encoding == Encoding.DENSE


def test_dense_fallback_accounts_for_block_offsets_and_uint32_indices():
    # Local16 needs a block offset table; for a tiny chunk this outweighs its narrower index.
    kwargs = dict(elements=4, width=2, changed=1, base_version=0, target_version=1)
    assert packet_header(**kwargs, preferred=Encoding.INDEX32).payload_bytes == 6
    assert packet_header(**kwargs, preferred=Encoding.BLOCK_LOCAL16).encoding == Encoding.DENSE


@pytest.fixture
def bits():
    # Signed zero and distinct NaN payloads must not disappear through float equality.
    base = torch.frombuffer(
        bytearray(struct.pack("<4I", 0, 0x7FC00001, 0x7F800000, 0xFF800000)), dtype=torch.int32
    ).clone()
    target = torch.frombuffer(
        bytearray(struct.pack("<4I", 0x80000000, 0x7FC00002, 0x7F800000, 0xFF800000)), dtype=torch.int32
    ).clone()
    header = packet_header(elements=4, width=4, changed=2, preferred=Encoding.INDEX32, base_version=2, target_version=7)
    return base, target, header


def test_receiver_and_sender_commit_exact_bits_only_after_verification_and_ack(bits):
    base, target, header = bits
    receiver, sender = ReplayBaseline(base.clone(), 2), ReplayBaseline(base.clone(), 2)
    receiver.validate_base(header, base)
    previous = receiver.install(header, target.clone(), target)
    assert torch.equal(previous, base) and receiver.version == 7 and torch.equal(receiver.values, target)
    # Receiver may be ahead until the ACK arrives; the sender must retain its old predecessor.
    assert sender.version == 2 and torch.equal(sender.values, base)
    sender.commit_after_ack(header, target, {"accepted": True, "base_version": 2, "target_version": 7})
    assert sender.version == 7 and torch.equal(sender.values, target)


@pytest.mark.parametrize("corrupted_index", [0, 2])
def test_full_base_check_catches_corruption_even_at_an_overwritten_index(bits, corrupted_index):
    base, _, header = bits
    corrupted = base.clone()
    corrupted[corrupted_index] ^= 1
    state = ReplayBaseline(corrupted, 2)
    with pytest.raises(ValueError, match="baseline differs"):
        state.validate_base(header, base)
    assert state.version == 2 and torch.equal(state.values, corrupted)


@pytest.mark.parametrize("mutate", ["target", "stale", "shape", "regression"])
def test_rejected_candidate_preserves_installed_bits_and_version(bits, mutate):
    base, target, header = bits
    state = ReplayBaseline(base.clone(), 2)
    if mutate == "target":
        staged = target.clone()
        staged[1] ^= 1
    else:
        staged = target.clone()
        header = {
            "stale": replace(header, base_version=0),
            "shape": replace(header, elements=5),
            "regression": replace(header, target_version=2),
        }[mutate]
    with pytest.raises(ValueError):
        state.install(header, staged, target)
    assert state.version == 2 and torch.equal(state.values, base)


@pytest.mark.parametrize(
    "ack",
    [
        {},
        {"accepted": False},
        {"accepted": True, "base_version": 0, "target_version": 7},
        {"accepted": True, "base_version": 2, "target_version": 8},
        {"accepted": 1, "base_version": 2, "target_version": 7},
        {"accepted": True, "base_version": 2.0, "target_version": 7},
        {"accepted": True, "base_version": 2, "target_version": 7.0},
    ],
)
def test_bad_or_lost_ack_fails_closed_without_advancing_sender(bits, ack):
    base, target, header = bits
    sender = ReplayBaseline(base.clone(), 2)
    with pytest.raises(ValueError, match="ACK"):
        sender.commit_after_ack(header, target, ack)
    assert sender.version == 2 and torch.equal(sender.values, base)


def test_receiver_rejects_duplicate_patch_instead_of_claiming_retry_recovery(bits):
    base, target, header = bits
    receiver = ReplayBaseline(base.clone(), 2)
    receiver.install(header, target.clone(), target)
    with pytest.raises(ValueError, match="version"):
        receiver.validate_base(header, base)
    assert receiver.version == 7 and torch.equal(receiver.values, target)


def test_float_equal_reference_cannot_stand_in_for_raw_bit_identity(bits):
    base, target, header = bits
    state = ReplayBaseline(base.clone(), 2)
    with pytest.raises(ValueError, match="baseline differs"):
        state.validate_base(header, base.to(torch.int64))
    with pytest.raises(ValueError, match="target differs"):
        state.install(header, target.clone(), target.to(torch.int64))
    assert state.version == 2 and torch.equal(state.values, base)


def test_receiver_requires_separate_staging_storage_even_for_noop(bits):
    base, _, header = bits
    state = ReplayBaseline(base.clone(), 2)
    with pytest.raises(ValueError, match="target differs"):
        state.install(header, state.values.view_as(state.values), base)
    assert state.version == 2 and torch.equal(state.values, base)
