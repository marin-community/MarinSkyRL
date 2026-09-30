"""Read-only taps inside a compiled vLLM engine's worker, installed as a vLLM worker extension.

The attention op is the one op that runs outside vLLM's piecewise CUDA graphs, so wrapping
``FlashAttentionImpl.forward`` sees every layer's query, key and value as the compiled pieces left
them and the output FA3 wrote, without changing what runs. ``EngineTaps`` is passed as the engine's
``worker_extension_cls`` (as SkyRL passes its own ``WorkerWrap``) and its methods are called by name
through ``LLM.collective_rpc``.
"""

from __future__ import annotations

import gc
import inspect
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

EXTENSION = "skyrl_train.mismatch_harness.engine_taps.EngineTaps"

_ATTENTION_CALLS: list[dict[str, Any]] = []


def attention_call(arguments: Mapping[str, Any], tokens: int) -> dict[str, Any]:
    """One FA3 call's real-token inputs and output (CPU copies) and the arguments that pick its kernel."""
    impl, metadata, kv_cache = arguments["self"], arguments["attn_metadata"], arguments["kv_cache"]
    key_cache = canonicalize_singleton_dim_strides(kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)[0])
    return {
        "layer_name": arguments["layer"].layer_name,
        "query": arguments["query"][:tokens].cpu(),
        "key": arguments["key"][:tokens].cpu(),
        "value": arguments["value"][:tokens].cpu(),
        "output": arguments["output"][:tokens].cpu(),
        "query_rows": arguments["query"].shape[0],
        "key_cache_stride": list(key_cache.stride()),
        "block_size": key_cache.shape[1],
        "sliding_window": list(impl.sliding_window),
        "scale": impl.scale,
        "softcap": impl.logits_soft_cap,
        "fa_version": impl.vllm_flash_attn_version,
        "max_query_len": metadata.max_query_len,
        "max_seq_len": metadata.max_seq_len,
        "seq_lens": metadata.seq_lens.tolist(),
        "scheduler_metadata": metadata.scheduler_metadata is not None,
        "max_num_splits": metadata.max_num_splits,
        "causal": str(metadata.causal),
        "use_cascade": metadata.use_cascade,
    }


def inductor_kernel_configs(objects: Iterable[object]) -> dict[str, list[str]]:
    """The launch configs each loaded Inductor Triton kernel holds, keyed by its source file name.

    A kernel holds every candidate config until its first launch; autotuning, or a ``.best_config``
    file found at load, leaves the one it launches. The file name is a hash of the kernel's source.
    """
    configs: dict[str, set[str]] = {}
    for obj in objects:
        if isinstance(obj, CachingAutotuner):
            launched = " | ".join(str(launcher.config) for launcher in obj.launchers)
            configs.setdefault(Path(str(obj.filename)).name, set()).add(launched)
    return {name: sorted(values) for name, values in configs.items()}


class EngineTaps:
    """Worker extension: record FA3 calls on one step, then save them with the model's kernel configs."""

    def tap_attention(self, step_tokens: int) -> None:
        """Keep what reaches every FA3 call on a step of ``step_tokens`` scheduled tokens."""
        original = FlashAttentionImpl.forward
        signature = inspect.signature(original)

        def forward(*args, **kwargs):
            output = original(*args, **kwargs)
            arguments = signature.bind(*args, **kwargs).arguments
            metadata = arguments["attn_metadata"]
            if metadata is not None and metadata.num_actual_tokens == step_tokens:
                _ATTENTION_CALLS.append(attention_call(arguments, step_tokens))
            return output

        FlashAttentionImpl.forward = forward

    def save_taps(self, path: str, positions: int) -> None:
        """Save the recorded FA3 calls, the model's rotary tables and every loaded kernel's configs."""
        tables = [buffer for name, buffer in self.get_model().named_buffers() if name.endswith("cos_sin_cache")]
        torch.save(
            {
                "attention": _ATTENTION_CALLS,
                "cos_sin_cache": [table[:positions].cpu() for table in tables],
                "kernel_configs": inductor_kernel_configs(gc.get_objects()),
            },
            path,
        )
