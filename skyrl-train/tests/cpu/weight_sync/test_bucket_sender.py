from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.bucket_sender import StreamingBucketSender
from skyrl_train.weight_sync.manifest import TensorSpec, build_manifest, pack_bucket


@pytest.fixture
def cuda_boundary(monkeypatch):
    log = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            log.append(("wait", self.name, event.number))

    class Event:
        number = 0

        def __init__(self):
            self.number = Event.number
            Event.number += 1
            self.joined = False

        def record(self, stream):
            self.joined = False
            log.append(("record", stream.name, self.number))

        def synchronize(self):
            self.joined = True
            log.append(("join", self.number))

        def query(self):
            return self.joined

    packs, caller = [], Stream("caller")

    def make_pack(device):
        pack = Stream("pack" if not packs else f"pack{len(packs)}")
        packs.append(pack)
        return pack

    @contextmanager
    def stream_context(stream):
        assert stream in packs
        yield

    monkeypatch.setattr(torch.cuda, "Stream", make_pack)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: caller)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    return SimpleNamespace(log=log, event_type=Event)


def fixture_parts():
    specs = [TensorSpec("experts.weight", (5, 2, 2), "bfloat16", True), TensorSpec("bias", (3,), "float32")]
    manifest = build_manifest(specs, bucket_bytes=24)
    sources = {
        "experts.weight": torch.arange(20, dtype=torch.bfloat16).reshape(5, 2, 2),
        "bias": torch.tensor([-0.0, float("nan"), 123.25], dtype=torch.float32),
    }
    buffers = (torch.empty(24, dtype=torch.uint8), torch.empty(24, dtype=torch.uint8))
    return manifest, sources, buffers


def test_streaming_export_matches_independent_bucket_bytes_and_joins_sends(cuda_boundary):
    manifest, sources, buffers = fixture_parts()
    emitted = []

    def export():
        for name, source in sources.items():
            emitted.append(name)
            yield name, source

    sender = StreamingBucketSender(manifest, export(), buffers)
    for bucket in range(manifest.bucket_count):
        actual = sender.pack_next_bucket()
        reference = torch.empty(24, dtype=torch.uint8)
        count = pack_bucket(manifest, bucket, sources, reference)
        assert torch.equal(actual, reference[:count])
        if bucket == 0:
            assert emitted == ["experts.weight"]
        sender.mark_bucket_sent(bucket)
    receipt = sender.finish()
    assert receipt["wire_bytes"] == 52
    assert receipt["source_count"] == 2
    assert receipt["send_completion_joined"]
    # Native CUDA boundary obligations: source->pack->send, then send->slot reuse.
    log = cuda_boundary.log
    assert log.index(("record", "caller", 4)) < log.index(("wait", "pack", 4))
    assert log.index(("record", "pack", 0)) < log.index(("wait", "caller", 0))
    assert log.index(("record", "caller", 2)) < log.index(("wait", "pack", 2))


def test_exporter_cannot_reuse_previous_source_before_pack_completion(cuda_boundary):
    manifest, sources, buffers = fixture_parts()

    def export():
        yield "experts.weight", sources["experts.weight"]
        assert cuda_boundary.log[-1] == ("join", 5)
        yield "bias", sources["bias"]
        assert cuda_boundary.log[-1] == ("join", 5)

    sender = StreamingBucketSender(manifest, export(), buffers)
    for bucket in range(manifest.bucket_count):
        sender.pack_next_bucket()
        sender.mark_bucket_sent(bucket)
    assert sender.finish()["source_count"] == 2


@pytest.mark.parametrize("failure", ["missing", "extra", "reordered", "dtype", "alias"])
def test_changed_export_rejects_without_identity_claim(cuda_boundary, failure):
    manifest, sources, buffers = fixture_parts()
    values = list(sources.items())
    if failure == "missing":
        values = values[:1]
    elif failure == "extra":
        values.append(("unexpected", torch.ones(1)))
    elif failure == "reordered":
        values.reverse()
    elif failure == "dtype":
        values[0] = (values[0][0], values[0][1].float())
    else:
        values[1] = ("bias", buffers[0][:12].view(torch.float32))
    sender = StreamingBucketSender(manifest, iter(values), buffers)
    with pytest.raises(ValueError):
        for bucket in range(manifest.bucket_count):
            sender.pack_next_bucket()
            sender.mark_bucket_sent(bucket)
        sender.finish()


def test_unsent_buffer_cannot_be_reused_or_declared_complete(cuda_boundary):
    manifest, sources, buffers = fixture_parts()
    sender = StreamingBucketSender(manifest, iter(sources.items()), buffers)
    sender.pack_next_bucket()
    with pytest.raises(ValueError, match="preceding bucket"):
        sender.pack_next_bucket()
    with pytest.raises(ValueError, match="exactly once"):
        sender.finish()
    with pytest.raises(ValueError, match="packed manifest order"):
        sender.mark_bucket_sent(1)


def test_pending_native_send_event_prevents_complete_receipt(cuda_boundary, monkeypatch):
    manifest, sources, buffers = fixture_parts()
    sender = StreamingBucketSender(manifest, iter(sources.items()), buffers)
    for bucket in range(manifest.bucket_count):
        sender.pack_next_bucket()
        sender.mark_bucket_sent(bucket)
    monkeypatch.setattr(cuda_boundary.event_type, "query", lambda event: False)
    with pytest.raises(RuntimeError, match="remains pending"):
        sender.finish()


@pytest.mark.parametrize("slot_count", [2, 3])
def test_ring_slots_preserve_bytes_across_wraparound_and_join_every_send(cuda_boundary, slot_count):
    tensor = torch.arange(40, dtype=torch.bfloat16).reshape(10, 2, 2)
    manifest = build_manifest([TensorSpec("weight", (10, 2, 2), "bfloat16", True)], bucket_bytes=16)
    buffers = tuple(torch.empty(16, dtype=torch.uint8) for _ in range(slot_count))
    sender = StreamingBucketSender(manifest, iter([("weight", tensor)]), buffers)
    chunks = []
    for bucket in range(manifest.bucket_count):
        chunks.append(sender.pack_next_bucket().clone())
        sender.mark_bucket_sent(bucket)
    receipt = sender.finish()
    assert torch.equal(torch.cat(chunks), tensor.view(torch.uint8).flatten())
    assert receipt["wire_bytes"] == 80 and receipt["send_completion_joined"]
    assert len({id(stream) for stream in sender.pack_streams}) == slot_count
    for slot in range(1, slot_count):
        assert ("wait", f"pack{slot}", slot - 1) in cuda_boundary.log
    for event in range(slot_count, 2 * slot_count):
        assert ("join", event) in cuda_boundary.log
    for reused_slot in range(min(slot_count, manifest.bucket_count - slot_count)):
        event = slot_count + reused_slot
        assert cuda_boundary.log.index(("record", "caller", event)) < cuda_boundary.log.index(
            ("wait", "pack" if reused_slot == 0 else f"pack{reused_slot}", event)
        )


def test_three_slot_ring_rejects_alias_of_last_slot(cuda_boundary):
    manifest, sources, buffers = fixture_parts()
    with pytest.raises(ValueError, match="must not alias"):
        StreamingBucketSender(manifest, iter(sources.items()), (*buffers, buffers[1]))
