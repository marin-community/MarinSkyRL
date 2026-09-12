"""Gather frozen source views directly into existing replay transfer buffers."""

import torch

from skyrl_train.weight_sync.bucket_sender import BucketSenderSlots
from skyrl_train.weight_sync.frozen_source_views import source_view


class FrozenViewBucketSender(BucketSenderSlots):
    def __init__(self, manifest, plan, local_sources, buffers):
        super().__init__(manifest, buffers)
        if len(plan) != manifest.bucket_count:
            raise ValueError("Frozen source schedule differs from the manifest")
        self.plan = plan
        self.local_sources = local_sources
        self.rank = torch.distributed.get_rank()
        self.source_ready = torch.cuda.Event()
        self.source_collectives = 0

    def pack_next_bucket(self):
        bucket, buffer = self.begin_bucket()
        # All source model work completed before entering the frozen interval;
        # this explicit event also orders any prior work on the caller's stream.
        self.source_ready.record(torch.cuda.current_stream(self.device))
        self.pack_stream.wait_event(self.source_ready)
        with torch.cuda.stream(self.pack_stream):
            for part in self.plan[bucket]:
                width = 2 if part.source.wire_dtype == "bfloat16" else 4
                destination = buffer.narrow(0, part.bucket_offset, part.source.numel * width)
                if self.rank == part.owner_rank:
                    source = source_view(part.source, self.local_sources)
                    if source.device != buffer.device or any(
                        source.untyped_storage().data_ptr() == slot.untyped_storage().data_ptr()
                        for slot in self.buffers
                    ):
                        raise ValueError("Frozen source must be independent storage on the local sender device")
                    destination.copy_(source.view(torch.uint8))
                torch.distributed.broadcast(destination, src=part.owner_rank)
                self.source_collectives += 1
        entry = self.manifest.bucket(bucket)[-1]
        return self.ready_bucket(entry.offset + entry.nbytes)

    def finish(self):
        self.finish_slots()
        self.finished = True
        return {
            "manifest_id": self.manifest.manifest_id,
            "bucket_count": self.manifest.bucket_count,
            "wire_bytes": sum(entry.nbytes for entry in self.manifest.entries),
            "source_collectives": self.source_collectives,
            "source_access": "original_parameter_views_into_existing_transfer_buffers",
            "send_completion_joined": True,
        }
