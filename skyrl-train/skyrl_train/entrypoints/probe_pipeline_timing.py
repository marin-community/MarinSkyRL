"""Two-GPU synthetic validation of production bucket streams and CUDA stage clocks.

Run with torchrun --standalone --nproc-per-node=2. No model or dataset is read.
Dense synthetic parameters exercise the production receiver copy path; this is
not evidence for expert layouts, cross-node links, or Snowball throughput.
"""

import argparse
from datetime import timedelta
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.bucket_identity import bucket_identity
from skyrl_train.weight_sync.bucket_receiver import GrugBucketReceiver
from skyrl_train.weight_sync.bucket_sender import StreamingBucketSender
from skyrl_train.weight_sync.manifest import TensorSpec, build_manifest
from skyrl_train.weight_sync.pipeline_timing import CudaPipelineTiming, overlap_seconds
from skyrl_train.weight_sync.worker_bucket_protocol import finish_worker_install, receive_worker_bucket


BUCKET_BYTES = 256 * 2**20
BUCKET_COUNT = 4


def validate_pair(rows):
    """Reject incomplete transfers or clocks before accepting a positive control."""
    assert len(rows) == 4
    assert {(row["mode"], row["rank"]) for row in rows} == {
        (mode, rank) for mode in ("serial", "pipeline") for rank in (0, 1)
    }
    assert len({row["manifest_id"] for row in rows}) == 1
    for row in rows:
        assert row["operation_complete"] and row["timing"]["events_complete"]
        stages = row["timing"]["intervals"]
        required = ("export", "pack", "nccl_send") if row["rank"] == 0 else ("nccl_receive", "load")
        assert set(stages) == set(required)
        assert all(len(stages[stage]) == BUCKET_COUNT for stage in required)
        for values in stages.values():
            overlap_seconds(values, values)  # Also reject malformed/nonfinite clocks.
        if row["rank"] == 1:
            assert row["installed_bytes"] == BUCKET_BYTES * BUCKET_COUNT
            assert row["mismatched_bytes"] == 0
        overlap = overlap_seconds(stages.get("export", []), stages.get("nccl_send", []))
        if row["rank"] == 0:
            if row["mode"] == "serial":
                assert overlap == 0, "Serial control must serialize export and send"
            else:
                assert overlap > 0, "Concurrent export/send control did not overlap"
    assert len({row["identity"]["gpu_uuid"] for row in rows}) == 2


def run_case(mode, device):
    rank = dist.get_rank()
    specs = [TensorSpec(f"weight_{i}", (BUCKET_BYTES // 2,), "bfloat16") for i in range(BUCKET_COUNT)]
    manifest = build_manifest(specs, bucket_bytes=BUCKET_BYTES)
    buffers = tuple(torch.empty(BUCKET_BYTES, dtype=torch.uint8, device=device) for _ in range(2))
    parameters = (
        {spec.name: torch.empty(spec.shape, dtype=torch.bfloat16, device=device) for spec in specs} if rank else {}
    )
    dist.barrier()
    torch.cuda.synchronize(device)
    timing = CudaPipelineTiming(device)
    mismatched_bytes = installed_bytes = 0
    if rank == 0:

        def exports():
            for i, spec in enumerate(specs):
                # Deliberate single-SM device work gives a measurable concurrent
                # stage without a host sleep or model-specific computation.
                torch.cuda._sleep(20_000_000)
                yield spec.name, torch.full(spec.shape, i + 1, dtype=torch.bfloat16, device=device)

        sender = StreamingBucketSender(
            manifest,
            exports(),
            buffers,
            export_stream=torch.cuda.Stream(device=device) if mode == "pipeline" else None,
            stage_timing=timing,
        )
        for bucket in range(manifest.bucket_count):
            buffer = sender.pack_next_bucket()
            stream = torch.cuda.current_stream(device)
            token = timing.start("nccl_send", stream)
            dist.broadcast(buffer, 0)
            timing.end(token, stream)
            sender.mark_bucket_sent(bucket)
            if mode == "serial":
                stream.synchronize()
        sender.finish()
        receipt = timing.finish()
    else:
        receiver = GrugBucketReceiver(manifest, parameters, {}, buffers, backend="TRITON", tensor_parallel_size=1)
        state = {
            "receiver": receiver,
            "publication_id": 1,
            "install_complete": False,
            "scratch": None,
            "slot_used": [False, False],
            "load_stream": torch.cuda.Stream(device=device),
            "receive_events": [torch.cuda.Event() for _ in buffers],
            "load_events": [torch.cuda.Event() for _ in buffers],
            "stage_timing": timing,
            "install_allocated_before": torch.cuda.memory_allocated(device),
            "install_free_before": torch.cuda.mem_get_info(device)[0],
        }
        worker = SimpleNamespace(device=device, _model_update_group=dist.group.WORLD, _diagnostic_bucket_state=state)
        for bucket in range(manifest.bucket_count):
            row = receive_worker_bucket(worker, bucket, manifest_id=manifest.manifest_id, publication_id=1)
            installed_bytes += row["installed_bytes"]
        finished = finish_worker_install(worker, manifest_id=manifest.manifest_id, publication_id=1)
        assert finished["completed_slots"] == [0, 1]
        receipt = finished["cuda_pipeline_stages"]
        # Full byte check after the production load-event completion join.
        for i, spec in enumerate(specs):
            expected = torch.tensor([i + 1], dtype=torch.bfloat16, device=device).view(torch.uint8)
            actual = parameters[spec.name].view(torch.uint8).view(-1, 2)
            mismatched_bytes += int(torch.count_nonzero(actual != expected).item())
    complete = torch.cuda.Event()
    complete.record(torch.cuda.current_stream(device))
    complete.synchronize()
    assert complete.query()
    return {
        "mode": mode,
        "rank": rank,
        "manifest_id": manifest.manifest_id,
        "identity": bucket_identity(device),
        "timing": receipt,
        "installed_bytes": installed_bytes,
        "mismatched_bytes": mismatched_bytes,
        "operation_complete": complete.query(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    assert dist.get_world_size() == 2 and torch.cuda.device_count() == 2
    try:
        # Warm the real group before either observed case.
        warm = torch.ones(1024, device=device)
        dist.broadcast(warm, 0)
        torch.cuda.synchronize(device)
        rows = []
        for mode in ("serial", "pipeline"):
            local = run_case(mode, device)
            gathered = [None, None]
            dist.all_gather_object(gathered, local)
            rows.extend(gathered)
        if rank == 0:
            print(
                "K9_DEVICE_TIMING_RECEIPT " + json.dumps({"source_commit": args.source_commit, "rows": rows}),
                flush=True,
            )
            validate_pair(rows)
            print(
                "K9_DEVICE_TIMING_PASS gpus=2 serial_export_send_zero=true pipeline_export_send_positive=true exact_bytes=true operation_complete=true",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
