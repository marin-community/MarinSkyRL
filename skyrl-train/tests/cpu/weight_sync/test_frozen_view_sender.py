from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from skyrl_train.weight_sync.frozen_source_plan import frozen_source_plan
from skyrl_train.weight_sync.cpu_source_catalogue import gather_source_catalogue
from skyrl_train.weight_sync.frozen_view_sender import FrozenViewBucketSender
from skyrl_train.weight_sync.manifest import pack_bucket
from tests.cpu.weight_sync.test_frozen_source_plan import plan_fixture


class CpuStream:
    def wait_event(self, event):
        assert event.recorded


class CpuEvent:
    def __init__(self):
        self.recorded = False

    def record(self, stream):
        self.recorded = True

    def synchronize(self):
        assert self.recorded

    def query(self):
        return self.recorded


def distributed_sender(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.cuda, "Stream", lambda device: CpuStream())
            patch.setattr(torch.cuda, "Event", CpuEvent)
            patch.setattr(torch.cuda, "current_stream", lambda device: CpuStream())
            patch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
            case = plan_fixture()
            gathered = gather_source_catalogue(case.rows[rank])
            plan = frozen_source_plan(case.manifest, gathered)
            buffers = (torch.empty(24, dtype=torch.uint8), torch.empty(24, dtype=torch.uint8))
            sender = FrozenViewBucketSender(case.manifest, plan, case.sources[rank], buffers)
            digest = hashlib.sha256()
            for bucket in range(case.manifest.bucket_count):
                actual = sender.pack_next_bucket()
                reference = torch.empty(24, dtype=torch.uint8)
                count = pack_bucket(case.manifest, bucket, case.full, reference)
                assert torch.equal(actual, reference[:count])
                digest.update(actual.numpy().tobytes())
                sender.mark_bucket_sent(bucket)
            receipt = sender.finish()
            receipt.update(rank=rank, sha256=digest.hexdigest(), mismatches=0)
            Path(output, f"rank-{rank}.json").write_text(json.dumps(receipt))
    finally:
        dist.destroy_process_group()


def test_actual_two_rank_gloo_replay_gathers_original_views_into_existing_buffers(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=distributed_sender, args=(rank, str(tmp_path / "rendezvous"), str(tmp_path)))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=25)
            assert not process.is_alive() and process.exitcode == 0
        receipts = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
        assert receipts[0]["sha256"] == receipts[1]["sha256"]
        assert all(
            row["wire_bytes"] == 40 and row["mismatches"] == 0 and row["source_collectives"] == 5 for row in receipts
        )
        assert all(row["send_completion_joined"] for row in receipts)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
