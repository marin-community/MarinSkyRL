"""Opt-in, source-pinned environment control before NCCL initialization."""

import hashlib
import inspect
import json
import os
import socket

import torch.distributed as dist

WORKER_CLASS = "skyrl_train.inference_engines.vllm.invariant_worker.InvariantWeightSyncWorker"
OVERRIDE_SHA256 = "9d72dcaa33ab4436ee6f760268468d8e5229dd143919d9658d8f3e81ef9de080"
EXPECTED_ENVIRONMENT = {
    "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_LAUNCH_MODE": "GROUP",
    "NCCL_COLLNET_ENABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_P2P_NET_DISABLE": "1",
    "NCCL_MIN_NCHANNELS": "1",
    "NCCL_MAX_NCHANNELS": "1",
    "NCCL_PROTO": "Simple",
    "NCCL_ALGO": "allreduce:tree",
    "NCCL_NTHREADS": "1",
    "NCCL_SOCKET_NTHREADS": "1",
    "VLLM_USE_AOT_COMPILE": "0",
}


def apply_weight_sync_environment(
    enabled: bool, *, role: str, rank: int | None = None, local_rank: int | None = None
) -> None:
    """Apply the qualified override in this process; emit only a fixed nonsecret allowlist.

    The source pin intentionally rejects a changed vLLM implementation. This control
    changes communication environment variables, not trainer arithmetic kernels.
    """
    if not enabled:
        return
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise ValueError("weight_sync_invariant_env requires VLLM_BATCH_INVARIANT=1 in each worker")
    if dist.is_initialized():
        raise RuntimeError("weight sync environment must be applied before the default process group")
    from vllm.model_executor.layers.batch_invariant import override_envs_for_invariance

    source_sha256 = hashlib.sha256(inspect.getsource(override_envs_for_invariance).encode()).hexdigest()
    if source_sha256 != OVERRIDE_SHA256:
        raise RuntimeError("weight sync environment override source differs from the qualified pin")
    override_envs_for_invariance()
    values = {key: os.environ.get(key) for key in EXPECTED_ENVIRONMENT}
    if values != EXPECTED_ENVIRONMENT:
        raise RuntimeError("weight sync environment override did not install the expected allowlist")
    receipt = {
        "role": role,
        "rank": rank if rank is not None else int(os.environ["RANK"]),
        "local_rank": local_rank if local_rank is not None else int(os.environ["LOCAL_RANK"]),
        "environment_rank": os.environ.get("RANK"),
        "environment_local_rank": os.environ.get("LOCAL_RANK"),
        "origin_host": socket.gethostname(),
        "origin_pid": os.getpid(),
        "default_process_group_initialized": dist.is_initialized(),
        "override_source_sha256": source_sha256,
        "values": values,
    }
    os.write(1, ("WEIGHT_SYNC_ENVIRONMENT_PRE_PG " + json.dumps(receipt, sort_keys=True) + "\n").encode())


def validate_weight_sync_environment_config(cfg) -> None:
    """Require both explicit control switches; reject unsupported worker selections."""
    enabled = cfg.trainer.algorithm.get("weight_sync_invariant_env", False)
    worker_class = (cfg.generator.get("engine_init_kwargs") or {}).get("worker_cls")
    if not enabled and worker_class != WORKER_CLASS:
        return
    if not enabled or worker_class != WORKER_CLASS:
        raise ValueError("weight_sync_invariant_env and its dedicated worker_cls must be enabled together")
    if cfg.generator.backend != "vllm" or not cfg.generator.run_engines_locally:
        raise ValueError("weight_sync_invariant_env requires local vLLM engines")
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise ValueError("weight_sync_invariant_env requires VLLM_BATCH_INVARIANT=1")
