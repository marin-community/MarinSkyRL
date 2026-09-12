"""Streaming complete HF tensors into bounded diagnostic weight-sync buffers.

The caller owns the learner's frozen interval. Timed installation retains one
complete exported tensor at a time; bounded replay uses original source views.
Neither path requests incomplete bridge conversion groups.
"""

from collections.abc import Iterator

import torch

from skyrl_train.weight_sync.manifest import PublicationManifest


class BucketSenderSlots:
    """Order source production, packing and transport on explicit CUDA streams."""

    def __init__(
        self,
        manifest: PublicationManifest,
        buffers: tuple[torch.Tensor, torch.Tensor],
    ):
        if not manifest.entries or len(buffers) != 2:
            raise ValueError("Streaming sender requires a nonempty manifest and two buffers")
        device = buffers[0].device
        for buffer in buffers:
            if (
                buffer.dtype != torch.uint8
                or buffer.shape != (manifest.bucket_bytes,)
                or not buffer.is_contiguous()
                or buffer.device != device
            ):
                raise ValueError("Sender buffers must match the manifest capacity and device")
        if buffers[0].untyped_storage().data_ptr() == buffers[1].untyped_storage().data_ptr():
            raise ValueError("Sender buffers must not alias")
        self.manifest = manifest
        self.buffers = buffers
        self.device = device
        self.pack_stream = torch.cuda.Stream(device=device)
        self.pack_events = (torch.cuda.Event(), torch.cuda.Event())
        self.sent_events = (torch.cuda.Event(), torch.cuda.Event())
        self.next_bucket = 0
        self.sent_bucket = -1
        self.finished = False

    def begin_bucket(self):
        if self.finished or self.next_bucket >= self.manifest.bucket_count:
            raise ValueError("No remaining manifest bucket")
        if self.sent_bucket != self.next_bucket - 1:
            raise ValueError("The preceding bucket must be submitted before packing another")
        bucket = self.next_bucket
        slot = bucket % 2
        if bucket >= 2:
            self.pack_stream.wait_event(self.sent_events[slot])
        buffer = self.buffers[slot]
        return bucket, buffer

    def ready_bucket(self, nbytes: int):
        slot = self.next_bucket % 2
        self.pack_events[slot].record(self.pack_stream)
        torch.cuda.current_stream(self.device).wait_event(self.pack_events[slot])
        self.next_bucket += 1
        return self.buffers[slot].narrow(0, 0, nbytes)

    def mark_bucket_sent(self, bucket: int):
        """Record after broadcast is enqueued on the caller's current stream."""
        if type(bucket) is not int or bucket != self.sent_bucket + 1 or bucket != self.next_bucket - 1:
            raise ValueError("Send acknowledgements must follow the packed manifest order")
        self.sent_events[bucket % 2].record(torch.cuda.current_stream(self.device))
        self.sent_bucket = bucket

    def finish_slots(self):
        if self.finished or self.sent_bucket != self.manifest.bucket_count - 1:
            raise ValueError("Every manifest bucket must be sent exactly once")
        for event in self.sent_events[: min(2, self.manifest.bucket_count)]:
            event.synchronize()
            if not event.query():
                raise RuntimeError("Sender transfer completion event remains pending")


class StreamingBucketSender(BucketSenderSlots):
    """Pack complete Bridge exports without retaining the complete HF model."""

    def __init__(self, manifest: PublicationManifest, sources: Iterator[tuple[str, torch.Tensor]], buffers):
        super().__init__(manifest, buffers)
        self.sources = iter(sources)
        self.source_ready = torch.cuda.Event()
        self.source_copied = torch.cuda.Event()
        self.current_name = None
        self.current_source = None
        self.source_count = 0

    def _next_source(self, entry):
        if self.current_source is not None:
            # A bridge generator may release or reuse its previous conversion
            # storage as soon as next() is called, on a different CUDA stream.
            self.source_copied.synchronize()
            self.current_source = None
        try:
            name, source = next(self.sources)
        except StopIteration as error:
            raise ValueError("Export ended before the complete manifest") from error
        if (
            name != entry.hf_name
            or tuple(source.shape) != entry.full_shape
            or source.dtype != getattr(torch, entry.wire_dtype)
            or not source.is_contiguous()
            or source.device != self.device
        ):
            raise ValueError("Export tensor order, shape, dtype or device differs from manifest")
        if any(source.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr() for buffer in self.buffers):
            raise ValueError("Export source must not alias either transfer buffer")
        self.current_name = name
        self.current_source = source
        self.source_count += 1
        self.source_ready.record(torch.cuda.current_stream(self.device))
        self.pack_stream.wait_event(self.source_ready)

    def pack_next_bucket(self):
        bucket, buffer = self.begin_bucket()
        entries = self.manifest.bucket(bucket)
        for entry in entries:
            if self.current_name != entry.hf_name:
                self._next_source(entry)
            with torch.cuda.stream(self.pack_stream):
                source = self.current_source.view(-1).narrow(0, entry.tensor_offset, entry.numel).view(torch.uint8)
                buffer.narrow(0, entry.offset, entry.nbytes).copy_(source)
                self.source_copied.record(self.pack_stream)
        return self.ready_bucket(entries[-1].offset + entries[-1].nbytes)

    def finish(self):
        """Join both send slots and reject missing or unexpected exported tensors."""
        self.finish_slots()
        self.source_copied.synchronize()
        self.current_source = None
        if next(self.sources, None) is not None:
            raise ValueError("Export contains a tensor absent from the manifest")
        self.finished = True
        return {
            "manifest_id": self.manifest.manifest_id,
            "bucket_count": self.manifest.bucket_count,
            "source_count": self.source_count,
            "wire_bytes": sum(entry.nbytes for entry in self.manifest.entries),
            "send_completion_joined": True,
        }
