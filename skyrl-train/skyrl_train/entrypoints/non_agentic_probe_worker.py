"""Worker-side readback for the K16 probe over vLLM's plain-data utility codec.

The pinned vLLM encodes utility RPC arguments with msgspec and, without
VLLM_ALLOW_INSECURE_SERIALIZATION, refuses any function object. The probe
therefore sends a `module:qualname` spec naming one allowed readback, and this
extension resolves it inside the worker. Results stay plain ints and strings.
"""

import importlib
import os

import torch

WORKER_MEMORY = "skyrl_train.entrypoints.non_agentic_probe_worker:worker_memory"
READBACKS = frozenset({WORKER_MEMORY})
# vLLM resolves worker_extension_cls with rsplit("."), unlike logits_processors.
PROBE_WORKER_EXTENSION = "skyrl_train.entrypoints.non_agentic_probe_worker.ProbeWorkerExtension"


def worker_memory(worker) -> dict:
    """Read actual allocated, reserved and free CUDA bytes inside each worker."""
    free, total = torch.cuda.mem_get_info()
    return {
        "pid": os.getpid(),
        "device": str(worker.device),
        "gpu_uuid": str(torch.cuda.get_device_properties(worker.device).uuid),
        "data_parallel_rank": worker.vllm_config.parallel_config.data_parallel_rank,
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "free": free,
        "total": total,
    }


def resolve_readback(spec: str):
    if spec not in READBACKS:
        raise ValueError(f"Unknown probe readback {spec!r}")
    module_path, qualname = spec.split(":")
    return getattr(importlib.import_module(module_path), qualname)


class ProbeWorkerExtension:
    """Mixed into the vLLM worker class through worker_extension_cls."""

    def read_probe_worker(self, spec: str) -> dict:
        return resolve_readback(spec)(self)
