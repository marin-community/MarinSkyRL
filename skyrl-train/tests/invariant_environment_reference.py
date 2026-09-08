"""Exact vLLM fa50698a9a30 override; Apache-2.0, vllm/model_executor/layers/batch_invariant.py."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.worker_base import WorkerBase


# fmt: off
def override_envs_for_invariance():
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # NCCL determinism settings
    os.environ["NCCL_LAUNCH_MODE"] = "GROUP"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["NCCL_P2P_NET_DISABLE"] = "1"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    os.environ["NCCL_NTHREADS"] = "1"
    os.environ["NCCL_SOCKET_NTHREADS"] = "1"

    # torch.compile settings
    os.environ["VLLM_USE_AOT_COMPILE"] = "0"
# fmt: on


def resolve_extended_worker(parallel_config, resolve_obj_by_qualname, logger):
    """Exact pinned worker resolution/extension block, with injected import/logger I/O."""
    if isinstance(parallel_config.worker_cls, str):
        worker_class: type[WorkerBase] = resolve_obj_by_qualname(parallel_config.worker_cls)
    else:
        raise ValueError(
            "passing worker_cls is no longer supported. "
            "Please pass keep the class in a separate module "
            "and pass the qualified name of the class as a string."
        )

    if parallel_config.worker_extension_cls:
        worker_extension_cls = resolve_obj_by_qualname(parallel_config.worker_extension_cls)
        extended_calls = []
        if worker_extension_cls not in worker_class.__bases__:
            # check any conflicts between worker and worker_extension_cls
            for attr in dir(worker_extension_cls):
                if attr.startswith("__"):
                    continue
                assert not hasattr(worker_class, attr), (
                    f"Worker class {worker_class} already has an attribute"
                    f" {attr}, which conflicts with the worker"
                    f" extension class {worker_extension_cls}."
                )
                if callable(getattr(worker_extension_cls, attr)):
                    extended_calls.append(attr)
            # dynamically inherit the worker extension class
            worker_class.__bases__ = worker_class.__bases__ + (worker_extension_cls,)
            logger.info(
                "Injected %s into %s for extended collective_rpc calls %s",
                worker_extension_cls,
                worker_class,
                extended_calls,
            )

    return worker_class


# Public class attributes from the pinned GPU Worker and WorkerBase (AST method inventory).
PINNED_WORKER_METHODS = (
    "__init__",
    "_check_weight_transfer_engine",
    "_maybe_get_memory_pool_context",
    "_scoped_allocator_max_split",
    "add_lora",
    "annotate_profile",
    "apply_model",
    "check_health",
    "compile_or_warm_up_model",
    "determine_available_memory",
    "elastic_ep_execute",
    "execute_dummy_batch",
    "execute_model",
    "finish_weight_update",
    "get_cache_block_size_bytes",
    "get_compilation_match_table",
    "get_encoder_timing_stats",
    "get_kv_cache_spec",
    "get_kv_connector_handshake_metadata",
    "get_model",
    "get_model_inspection",
    "get_supported_tasks",
    "init_device",
    "init_weight_transfer_engine",
    "initialize_from_config",
    "list_loras",
    "load_model",
    "pin_lora",
    "profile",
    "reload_weights",
    "remove_lora",
    "reset_encoder_cache",
    "reset_mm_cache",
    "sample_tokens",
    "save_sharded_state",
    "save_tensorized_model",
    "shutdown",
    "sleep",
    "start_weight_update",
    "take_draft_token_ids",
    "update_config",
    "update_max_model_len",
    "update_weights",
    "vocab_size",
    "wake_up",
)
