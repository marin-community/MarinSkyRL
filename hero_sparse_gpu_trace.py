"""Observe one warm update at the first and last pipeline stages."""

from pathlib import Path
import gzip
import hashlib
import json
import os
import tempfile


def record_cuda_call(native_call, arguments, trace_path):
    # Torch is an optional native dependency, imported only inside the worker.
    import torch

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as profiler:
        result = native_call(*arguments)
    profiler.export_chrome_trace(str(trace_path))
    return result


def training_step(worker, native_call, train_data, timing):
    # The counter belongs only to this observer, never to the optimizer.
    import torch
    from megatron.core import parallel_state
    from hero_qualification import s3_client, s3_location

    update = worker.__dict__.get("_qualification_trace_update", 0)
    worker._qualification_trace_update = update + 1
    rank = torch.distributed.get_rank()
    if update != 1 or rank not in (0, 44):
        return native_call(train_data, timing)
    assert torch.distributed.get_world_size() == 48
    assert parallel_state.get_pipeline_model_parallel_world_size() == 12
    assert parallel_state.get_pipeline_model_parallel_rank() == (0 if rank == 0 else 11)
    with tempfile.TemporaryDirectory(prefix="hero-sparse-trace-") as directory:
        trace_path = Path(directory) / f"rank-{rank}-warm-update-1.json"
        result = record_cuda_call(native_call, (train_data, timing), trace_path)
        compressed_path = trace_path.with_suffix(".json.gz")
        with trace_path.open("rb") as source, gzip.open(compressed_path, "wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
        bucket, prefix = s3_location(os.environ["HERO_OUTPUT"])
        key = f"{prefix}/sparse-gpu-traces/{compressed_path.name}"
        client = s3_client()
        client.upload_file(str(compressed_path), bucket, key)
        with compressed_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        summary = {
            "rank": rank,
            "pipeline_rank": parallel_state.get_pipeline_model_parallel_rank(),
            "update": update,
            "torch": torch.__version__,
            "trace_uri": f"s3://{bucket}/{key}",
            "trace_bytes": compressed_path.stat().st_size,
            "trace_sha256": digest,
            "scope": "One diagnostic warm update at two ranks. Profiling and upload perturb timing; not a throughput gate.",
        }
        client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/sparse-gpu-traces/rank-{rank}-manifest.json",
            Body=json.dumps(summary).encode(),
        )
        return result
